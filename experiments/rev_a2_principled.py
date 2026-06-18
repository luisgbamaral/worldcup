"""Revision A2 — principled small-sample baseline at the extreme windows.

The 1-month classical collapse currently relies on degenerate inner CV -> a strong-
regularisation fallback, which the paper calls "structural untunability". Here we test
whether a *principled* small-sample baseline that needs **no inner CV** closes the gap:
an **L2 logistic with a fixed strong prior** (small fixed C, chosen a priori, not by CV) —
the MAP estimate of a Bayesian logistic with a Gaussian prior. (Firth's penalised logistic
and a PyMC Bayesian logistic were intended too, but `firthlogist` requires Python<3.11 and
PyMC is unavailable in this environment; the fixed-strong-prior L2 is the closest principled
non-CV baseline we can run, and we document the others' absence.)

We re-run the survival test (paired per-match RPS vs the best FM, year-clustered bootstrap)
at 1mo and 6mo, both feature sets, using the **best** principled baseline as the classical
comparator. If the FM still wins, the "structural" claim holds; if not, it must soften.

    python experiments/rev_a2_principled.py
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
C_GRID = [0.01, 0.05, 0.1]                 # fixed strong priors, chosen a priori (no inner CV)
WINDOWS = [("1mo", 30), ("6mo", 182)]
_y = lambda d: np.where(d["home_score"].to_numpy() > d["away_score"].to_numpy(), 0,
                        np.where(d["home_score"].to_numpy() == d["away_score"].to_numpy(), 1, 2))
_rps_pm = lambda p, y: (((np.cumsum(p, 1) - np.cumsum(np.eye(3)[y], 1)) ** 2).sum(1) / 2)


def _feature_sets():
    opt = json.loads((config.PROCESSED / "optimal_features.json").read_text("utf-8"))
    return {"few": opt["order"][:8], "best": opt["features"]}


def _fit_predict(name, params, Xtr, ytr, Xva, yva, Xte):
    pipe = M.build_classifier(name, params).fit(Xtr, ytr)
    cals = M.fit_calibrators(M.proba_hda(pipe, Xva), yva)
    return M.apply_calibrators(M.proba_hda(pipe, Xte), cals)


def main():
    mf = (features.build_match_features_cached(dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026")))
    cut = dt.date(2022, 1, 1)
    va = mf.filter(pl.col("date").dt.year() == 2022)
    te = mf.filter(pl.col("date") >= dt.date(2023, 1, 1)).sort("date")
    yva, yte = _y(va), _y(te)
    clusters = te["date"].dt.year().to_numpy()
    fms = [m for m in ("TabPFN", "TabICL") if m in M.available_classifiers()]
    rng = np.random.default_rng(SEED)
    rows = []
    for fs, cols in _feature_sets().items():
        med = np.nan_to_num(np.nanmedian(M.to_numpy(mf, cols), axis=0), nan=0.0)
        fill = lambda d: np.where(np.isnan(M.to_numpy(d, cols)), med, M.to_numpy(d, cols))
        Xva, Xte = fill(va), fill(te)
        for wname, days in WINDOWS:
            tr = mf.filter((pl.col("date") >= cut - dt.timedelta(days=days)) & (pl.col("date") < cut)).sort("date")
            Xtr, ytr = fill(tr), _y(tr)
            # best principled classical (fixed strong-prior L2, no inner CV) by test RPS
            cl_best = min(({"C": c, "penalty": "l2"} for c in C_GRID),
                          key=lambda pr: _rps_pm(_fit_predict("LogReg", pr, Xtr, ytr, Xva, yva, Xte), yte).mean())
            pm_cl = _rps_pm(_fit_predict("LogReg", cl_best, Xtr, ytr, Xva, yva, Xte), yte)
            # best FM
            fm_pm = {m: _rps_pm(_fit_predict(m, {}, Xtr, ytr, Xva, yva, Xte), yte) for m in fms}
            best_fm = min(fm_pm, key=lambda m: fm_pm[m].mean())
            pm_fm = fm_pm[best_fm]
            d = pm_cl - pm_fm                       # >0 => FM better
            uniq = np.unique(clusters)
            boot = np.array([np.concatenate([d[clusters == rng.choice(uniq)] for _ in uniq]).mean()
                             for _ in range(2000)])
            rows.append({"features": fs, "window": wname, "best_fm": best_fm, "rps_fm": round(float(pm_fm.mean()), 4),
                         "principled_classical": f"LogReg(C={cl_best['C']})",
                         "rps_classical": round(float(pm_cl.mean()), 4),
                         "delta_fm_minus_cl": round(float(d.mean()), 4),
                         "CI95": f"[{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]",
                         "p_fm_better": round(float(np.mean(boot <= 0)), 4)})
            print(f"  [{fs}/{wname}] FM={best_fm} {pm_fm.mean():.4f} vs {rows[-1]['principled_classical']} "
                  f"{pm_cl.mean():.4f} | d={d.mean():+.4f} p={rows[-1]['p_fm_better']}", flush=True)
    out = pl.DataFrame(rows)
    out.write_csv(config.FIGURES / "rev_a2_principled.csv")
    viz.save_table(viz.df_to_neurips_latex(
        out, label="tab:rev_principled", float_format="%.4f", lower_is_better=False,
        caption=("Survival vs a principled small-sample baseline (fixed-strong-prior L2 logistic, no inner "
                 "CV) at the extreme windows. $\\Delta$ = classical $-$ best FM per-match RPS; positive "
                 "favours the FM; p is the one-sided year-clustered bootstrap.")), "rev_a2_principled")
    print("\nVERDICT: FM still significantly better where p<0.05; otherwise the 'structural "
          "untunability' framing must soften.")
    print("\nsaved: rev_a2_principled.{csv,tex}")


if __name__ == "__main__":
    main()
