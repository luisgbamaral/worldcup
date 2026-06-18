"""Revision A3 — TabPFN 10,000-row cap sensitivity on the whole series.

In the broad-series setting (train < 2022) the training set is ~20.8k rows, so TabPFN's
10k training-context cap **binds**. The choice of which 10k to keep is undocumented and
could silently drive the headline FM number. We compare two documented strategies:
  (i)  most-recent 10k (by date)
  (ii) random 10k (fixed seed)
and report whole-series RPS, log-loss and Brier for both, for TabPFN (and TabICL, which has
the same cap). Test = 2023+, calibration = 2022.

    python experiments/rev_a3_tabpfn_cap.py
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

SEED = M.SEED
CAP = 10000
_y = lambda d: np.where(d["home_score"].to_numpy() > d["away_score"].to_numpy(), 0,
                        np.where(d["home_score"].to_numpy() == d["away_score"].to_numpy(), 1, 2))


def _cols():
    p = config.PROCESSED / "optimal_features.json"
    mf = features.build_match_features_cached(dt.date(2000, 1, 1))
    return json.loads(p.read_text("utf-8"))["features"] if p.exists() else M.feature_columns(mf)


def _scores(p, y):
    Y = np.eye(3)[y]
    rps = float((((np.cumsum(p, 1) - np.cumsum(Y, 1)) ** 2).sum(1) / 2).mean())
    ll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    brier = float(((p - Y) ** 2).sum(1).mean())
    return rps, ll, brier


def main():
    mf = (features.build_match_features_cached(dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026")))
    cols = _cols()
    tr = mf.filter(pl.col("date") < dt.date(2022, 1, 1)).filter(pl.col("home_score").is_not_null()).sort("date")
    va = mf.filter(pl.col("date").dt.year() == 2022)
    te = mf.filter(pl.col("date") >= dt.date(2023, 1, 1))
    med = np.nan_to_num(np.nanmedian(M.to_numpy(tr, cols), axis=0), nan=0.0)
    fill = lambda d: np.where(np.isnan(M.to_numpy(d, cols)), med, M.to_numpy(d, cols))
    Xtr, Xva, Xte = fill(tr), fill(va), fill(te)
    ytr, yva, yte = _y(tr), _y(va), _y(te)
    binds = len(ytr) > CAP
    print(f"whole-series train rows={len(ytr)} | TabPFN/TabICL cap binds: {binds}")

    strategies = {"recent_10k": np.arange(len(ytr))[-CAP:],
                  "random_10k": np.random.RandomState(SEED).choice(len(ytr), CAP, replace=False)}
    rows = []
    for name in [m for m in ("TabPFN", "TabICL") if m in M.available_classifiers()]:
        for strat, idx in strategies.items():
            try:
                pipe = M.build_classifier(name).fit(Xtr[idx], ytr[idx])
                cals = M.fit_calibrators(M.proba_hda(pipe, Xva), yva)
                p = M.apply_calibrators(M.proba_hda(pipe, Xte), cals)
                rps, ll, brier = _scores(p, yte)
                rows.append({"model": name, "subsample": strat, "n_train": int(len(idx)),
                             "RPS": round(rps, 4), "logloss": round(ll, 4), "Brier": round(brier, 4)})
                print(f"  {name:7} {strat:11} RPS={rps:.4f} logloss={ll:.4f} Brier={brier:.4f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {name}/{strat}: {type(exc).__name__} {str(exc)[:60]}")
    out = pl.DataFrame(rows)
    out.write_csv(config.FIGURES / "rev_a3_tabpfn_cap.csv")
    # sensitivity = max - min RPS across strategies, per model
    sens = out.group_by("model").agg(RPS_spread=(pl.col("RPS").max() - pl.col("RPS").min()))
    print("\nRPS spread between subsampling strategies (sensitivity):")
    print(sens)
    viz.save_table(viz.df_to_neurips_latex(
        out, label="tab:rev_tabpfn_cap", float_format="%.4f",
        caption=("TabPFN/TabICL 10k training-cap sensitivity on the whole series (cap binds at "
                 f"{len(ytr)} rows): most-recent-10k vs random-10k context, with RPS/log-loss/Brier.")),
        "rev_a3_tabpfn_cap")
    print("\nsaved: rev_a3_tabpfn_cap.{csv,tex}")


if __name__ == "__main__":
    main()
