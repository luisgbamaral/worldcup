"""Forward feature-selection curve: RPS vs. number of features, per model.

Greedy forward selection (wrapper, guided by the cheap LogReg) builds ONE feature
order, forced to start from the simplest wins-based feature (``win_rate`` / ``ppg``).
Every model is then evaluated along that order at increasing prefix lengths, giving
an RPS-vs-#features curve per model — to see whether the tune-free foundation models
win with few features and where the optimum lies. The optimal subset is written out
so the World-Cup evaluation can be regenerated with it (replacing the aggressive CFS,
which collapsed to the two Elo columns).

Temporal split for the curve: train < 2022 (full history, subsampled), validate 2022,
test 2023+ — coherent with the one-step whole-series evaluation.

    python experiments/feature_forward.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldcup import config, features, modeling as M, viz  # noqa: E402

CHEAP = ["LogReg", "CatBoost", "XGBoost"]
FM_GRID = [1, 2, 3, 5, 8, 15, 30]            # prefix lengths at which the (cloud/CPU) FMs run
_y = lambda d: np.where(d["home_score"].to_numpy() > d["away_score"].to_numpy(), 0,
                        np.where(d["home_score"].to_numpy() == d["away_score"].to_numpy(), 1, 2))


def _data():
    mf = (features.build_match_features_cached(train_cut=dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026")))
    # rank features on the FULL training history (all eras) for coherence;
    # main() subsamples the rows for speed (a random sample keeps every era).
    tr = mf.filter(pl.col("date") < dt.date(2022, 1, 1))
    va = mf.filter(pl.col("date").dt.year() == 2022)
    te = mf.filter(pl.col("date") >= dt.date(2023, 1, 1))
    return mf, tr, va, te


def _imputed(tr, va, te, cols):
    med = np.nan_to_num(np.nanmedian(M.to_numpy(tr, cols), axis=0), nan=0.0)
    fill = lambda d: np.where(np.isnan(M.to_numpy(d, cols)), med, M.to_numpy(d, cols))
    return fill(tr), fill(va), fill(te)


def _rps(name, Xtr, ytr, Xva, yva, Xte, yte):
    pipe = M.build_classifier(name).fit(Xtr, ytr)
    cals = M.fit_calibrators(M.proba_hda(pipe, Xva), yva)
    return M.rps(M.apply_calibrators(M.proba_hda(pipe, Xte), cals), yte)


def _fast_lr(Xtr, ytr, Xva, yva, idx):
    """Lightweight LogReg val-RPS (no pipeline/calibration) — used only to rank features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler().fit(Xtr[:, idx])
    clf = LogisticRegression(max_iter=200).fit(s.transform(Xtr[:, idx]), ytr)
    p = clf.predict_proba(s.transform(Xva[:, idx]))
    return M.rps(p[:, [list(clf.classes_).index(i) for i in range(3)]], yva)


def wins_feature(Xtr, ytr, Xva, yva, cols):
    """Index of the best single wins-based feature — the forced 'model 0' reference."""
    win = [i for i, c in enumerate(cols) if "win_rate" in c or c.startswith("ppg")]
    return min(win, key=lambda i: _fast_lr(Xtr, ytr, Xva, yva, [i]))


def forward_order(Xtr, ytr, Xva, yva, cols, max_steps=45):
    """Greedy forward selection FROM EMPTY (every feature, incl. wins, competes).
    'model 0' (wins only) is a separate reference; model 1 = best single feature,
    model 2 = + next best, ... — LogReg-guided. Stops after ``max_steps`` picks
    (the optimum is well below that; avoids the O(n_features^2) full ordering)."""
    score = lambda idx: _fast_lr(Xtr, ytr, Xva, yva, idx)
    remaining, order = list(range(len(cols))), []
    while remaining and len(order) < max_steps:
        order.append(min(remaining, key=lambda j: score(order + [j])))
        remaining.remove(order[-1])
        if len(order) % 5 == 0:
            print(f"  greedy step {len(order):2}/{max_steps}  last+={cols[order[-1]]}", flush=True)
    return order


def main():
    mf, tr, va, te = _data()
    cols = M.feature_columns(tr)
    Xtr, Xva, Xte = _imputed(tr, va, te, cols)
    ytr, yva, yte = _y(tr), _y(va), _y(te)
    # rank/curve on a random subsample of the FULL history (every era represented), for speed
    if len(ytr) > 10000:
        sub = np.random.RandomState(M.SEED).choice(len(ytr), 10000, replace=False)
        Xtr, ytr = Xtr[sub], ytr[sub]
    print(f"features={len(cols)}  train={len(ytr)} (of {tr.height}) val={va.height} test={te.height}")

    m0 = wins_feature(Xtr, ytr, Xva, yva, cols)        # model 0: forced wins only
    order = forward_order(Xtr, ytr, Xva, yva, cols)     # model 1..N: greedy from empty
    print("model 0 (forced wins):", cols[m0])
    print("greedy order (model 1 -> ...):", [cols[i] for i in order[:12]])

    # step 0 = wins-only reference; step k>=1 = the greedy top-k (k free features);
    # step = len(cols) = the full feature set (upper reference, all columns)
    fms = [m for m in ("TabPFN", "TabICL", "TabDPT") if m in M.available_classifiers()]
    K = len(order)
    steps_cheap = sorted(set(list(range(1, K + 1)) + [len(cols)]))
    rows = []
    for name in CHEAP + fms:
        steps = [0] + (steps_cheap if name in CHEAP else
                       [k for k in FM_GRID if k <= K] + [len(cols)])
        for s in steps:
            idx = [m0] if s == 0 else (list(range(len(cols))) if s == len(cols) else order[:s])
            try:
                r = _rps(name, Xtr[:, idx], ytr, Xva[:, idx], yva, Xte[:, idx], yte)
                rows.append({"model": name, "step": s, "n_features": len(idx), "rps": r})
                print(f"  {name:9} step={s:3} ({len(idx)}f) RPS={r:.4f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {name} step={s}: {type(exc).__name__}")
    res = pl.DataFrame(rows)
    res.write_parquet(config.PROCESSED / "feature_forward.parquet")

    # optimal = greedy step (>=1) minimising pooled RPS across the cheap models
    pooled = (res.filter(pl.col("model").is_in(CHEAP) & (pl.col("step") >= 1))
              .group_by("step").agg(pl.col("rps").mean()).sort("rps"))
    k_star = int(pooled["step"][0])
    optimal = list(cols) if k_star == len(cols) else [cols[i] for i in order[:k_star]]
    (config.PROCESSED / "optimal_features.json").write_text(
        json.dumps({"k_star": k_star, "model0_wins": cols[m0], "features": optimal,
                    "order": [cols[i] for i in order]}, indent=2), encoding="utf-8")
    print(f"\nmodel 0 = {cols[m0]} (wins) | optimal k* = {k_star} -> {optimal}")

    viz.set_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(viz.TEXTWIDTH_IN, viz.TEXTWIDTH_IN * 0.6))
    for i, name in enumerate(sorted(res["model"].unique().to_list())):
        s = res.filter(pl.col("model") == name).sort("step")
        ax.plot(s["step"], s["rps"], label=name, **viz.style_for(i))
    ax.axvline(k_star, color="0.7", ls=":", lw=0.8)
    ax.set(xlabel="model index  (0 = wins-only; k = greedy top-k free features)",
           ylabel="test RPS (2022, lower is better)", title="Forward feature-selection curve")
    ax.legend(ncol=2, fontsize=6)
    viz.save_fig(fig, "13_feature_forward")
    res.sort("model", "step").write_csv(config.FIGURES / "feature_forward.csv")
    print("saved: 13_feature_forward.{pdf,png}, feature_forward.csv, optimal_features.json")


if __name__ == "__main__":
    main()
