"""Revision D2 — compute / wall-clock cost per method.

Times train + inference for every model on the whole-series setup (one fit, predict on the
2023+ test), and tabulates it next to the per-cell HPO cost. Substantiates the
"configuration-free / zero tuning cost" claim quantitatively. TabPFN/TabICL are cloud/CPU
foundation models with **0** tuning fits; the classical models pay an inner-CV HPO budget.

    python experiments/rev_d2_cost.py
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
from worldcup import config, features, modeling as M, viz  # noqa: E402

_y = lambda d: np.where(d["home_score"].to_numpy() > d["away_score"].to_numpy(), 0,
                        np.where(d["home_score"].to_numpy() == d["away_score"].to_numpy(), 1, 2))
# per-cell inner-CV HPO fit counts (from the ablation): LogReg 6x2 grid x3 folds, GBDT 10 trials x3 folds
HPO_FITS = {"LogReg": 36, "CatBoost": 30, "XGBoost": 30, "TabPFN": 0, "TabICL": 0}


def main():
    mf = (features.build_match_features_cached(dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026")))
    p = config.PROCESSED / "optimal_features.json"
    cols = json.loads(p.read_text("utf-8"))["features"] if p.exists() else M.feature_columns(mf)
    tr = mf.filter(pl.col("date") < dt.date(2022, 1, 1)).filter(pl.col("home_score").is_not_null()).sort("date")
    te = mf.filter(pl.col("date") >= dt.date(2023, 1, 1))
    med = np.nan_to_num(np.nanmedian(M.to_numpy(tr, cols), axis=0), nan=0.0)
    fill = lambda d: np.where(np.isnan(M.to_numpy(d, cols)), med, M.to_numpy(d, cols))
    Xtr, Xte, ytr = fill(tr), fill(te), _y(tr)
    rows = []
    for name in [m for m in ("LogReg", "CatBoost", "XGBoost", "TabPFN", "TabICL") if m in M.available_classifiers()]:
        Xf, yf = (Xtr, ytr)
        if name in ("TabPFN", "TabICL") and len(ytr) > 10000:
            sub = np.random.RandomState(M.SEED).choice(len(ytr), 10000, replace=False); Xf, yf = Xtr[sub], ytr[sub]
        t0 = time.time(); pipe = M.build_classifier(name).fit(Xf, yf); t_fit = time.time() - t0
        t0 = time.time(); _ = M.proba_hda(pipe, Xte); t_pred = time.time() - t0
        rows.append({"model": name, "n_train": int(len(yf)), "n_test": te.height,
                     "fit_s": round(t_fit, 2), "predict_s": round(t_pred, 2),
                     "hpo_fits_per_cell": HPO_FITS.get(name, 0),
                     "tune_free": name in ("TabPFN", "TabICL")})
        print(f"  {name:9} fit={t_fit:6.2f}s predict={t_pred:6.2f}s hpo_fits/cell={HPO_FITS.get(name,0)}", flush=True)
    out = pl.DataFrame(rows)
    out.write_csv(config.FIGURES / "rev_d2_cost.csv")
    print(out)
    viz.save_table(viz.df_to_neurips_latex(out, label="tab:rev_cost", float_format="%.2f",
        caption=("Compute cost per method on the whole series (one fit + inference on the 2023+ test). "
                 "hpo\\_fits\\_per\\_cell is the inner-CV budget in the training-size ablation; foundation "
                 "models are tune-free (0). TabPFN runs on the PriorLabs cloud, TabICL on CPU.")),
        "rev_d2_cost")
    print("\nsaved: rev_d2_cost.{csv,tex}")


if __name__ == "__main__":
    main()
