"""Revision B3 — rolling-origin multi-era backtest.

The whole-series null currently rests on a single 2023+ test era. Here we add several
non-overlapping, temporally ordered test eras and report per-era RPS for every model, to
see whether "LogReg best at full data" and the small FM edge are period-stable or
period-specific. Each era: train on the 8 years before it, calibrate on the prior year,
test on the era. Optimal (107) feature set; foundation models tune-free.

    python experiments/rev_b3_multiera.py
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
ERAS = [(2015, 2017), (2017, 2019), (2019, 2021), (2021, 2023), (2023, 2027)]  # [start, end)
MODELS = ["LogReg", "CatBoost", "XGBoost"]
_y = lambda d: np.where(d["home_score"].to_numpy() > d["away_score"].to_numpy(), 0,
                        np.where(d["home_score"].to_numpy() == d["away_score"].to_numpy(), 1, 2))


def _cols(mf):
    p = config.PROCESSED / "optimal_features.json"
    return json.loads(p.read_text("utf-8"))["features"] if p.exists() else M.feature_columns(mf)


def main():
    mf = (features.build_match_features_cached(dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026"))
          .filter(pl.col("home_score").is_not_null()))
    cols = _cols(mf)
    fms = [m for m in ("TabPFN", "TabICL") if m in M.available_classifiers()]
    rows = []
    for y0, y1 in ERAS:
        te = mf.filter((pl.col("date").dt.year() >= y0) & (pl.col("date").dt.year() < y1))
        va = mf.filter(pl.col("date").dt.year() == y0 - 1)
        tr = mf.filter((pl.col("date") < dt.date(y0 - 1, 1, 1))
                       & (pl.col("date") >= dt.date(y0 - 9, 1, 1))).sort("date")
        if te.is_empty() or tr.height < 500:
            continue
        med = np.nan_to_num(np.nanmedian(M.to_numpy(tr, cols), axis=0), nan=0.0)
        fill = lambda d: np.where(np.isnan(M.to_numpy(d, cols)), med, M.to_numpy(d, cols))
        Xtr, Xva, Xte = fill(tr), fill(va), fill(te)
        ytr, yva, yte = _y(tr), _y(va), _y(te)
        era = f"{y0}-{y1-1}"
        for name in MODELS + fms:
            try:
                Xf, yf = (Xtr, ytr)
                if name in ("TabPFN", "TabICL") and len(ytr) > 10000:
                    sub = np.random.RandomState(SEED).choice(len(ytr), 10000, replace=False)
                    Xf, yf = Xtr[sub], ytr[sub]
                pipe = M.build_classifier(name).fit(Xf, yf)
                cals = M.fit_calibrators(M.proba_hda(pipe, Xva), yva)
                p = M.apply_calibrators(M.proba_hda(pipe, Xte), cals)
                rows.append({"era": era, "n_test": te.height, "model": name,
                             "RPS": round(float(M.rps(p, yte)), 4)})
                print(f"  [{era}] {name:9} n={te.height:4} RPS={M.rps(p, yte):.4f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip {era}/{name}] {type(exc).__name__} {str(exc)[:50]}")
    out = pl.DataFrame(rows)
    out.write_csv(config.FIGURES / "rev_b3_multiera.csv")
    piv = out.pivot("era", index="model", values="RPS")
    print("\n=== RPS per era (rows=model) ===")
    print(piv)
    # rank stability: best model per era
    best = out.sort("RPS").group_by("era").first().select("era", best="model").sort("era")
    print("\nbest (lowest RPS) per era:")
    print(best)
    viz.save_table(viz.df_to_neurips_latex(piv, label="tab:rev_multiera", float_format="%.4f",
        caption="Rolling-origin multi-era backtest: one-step RPS per non-overlapping test era (train on the "
                "prior 8 years, calibrate on the prior year). Tests whether the model ordering is "
                "period-stable."), "rev_b3_multiera")
    print("\nsaved: rev_b3_multiera.{csv,tex}")


if __name__ == "__main__":
    main()
