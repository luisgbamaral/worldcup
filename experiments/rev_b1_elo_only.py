"""Revision B1 — Elo-only reference baseline.

A logistic regression on the single Elo-difference feature (`elo_diff_eff`), as a reference
row quantifying the incremental value of the full 107-feature table over the dominant
covariate. It is a reference line, not the significance reference (LogReg-on-all-features
stays that). Reported for the whole series (test 2023+) and the pooled World Cups.

    python experiments/rev_b1_elo_only.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldcup import config, features, modeling as M, viz  # noqa: E402

FEAT = ["elo_diff_eff"]
_y = lambda d: np.where(d["home_score"].to_numpy() > d["away_score"].to_numpy(), 0,
                        np.where(d["home_score"].to_numpy() == d["away_score"].to_numpy(), 1, 2))


def _scores(p, y):
    Y = np.eye(3)[y]
    return (float((((np.cumsum(p, 1) - np.cumsum(Y, 1)) ** 2).sum(1) / 2).mean()),
            float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
            float(((p - Y) ** 2).sum(1).mean()), float((p.argmax(1) == y).mean()))


def _fit(tr, va, te):
    X = lambda d: np.nan_to_num(M.to_numpy(d, FEAT))
    pipe = M.build_classifier("LogReg").fit(X(tr), _y(tr))
    cals = M.fit_calibrators(M.proba_hda(pipe, X(va)), _y(va))
    return M.apply_calibrators(M.proba_hda(pipe, X(te)), cals)


def main():
    mf = (features.build_match_features_cached(dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026"))
          .filter(pl.col("home_score").is_not_null()))
    rows = []
    # whole series
    tr = mf.filter(pl.col("date") < dt.date(2022, 1, 1))
    va = mf.filter(pl.col("date").dt.year() == 2022)
    te = mf.filter(pl.col("date") >= dt.date(2023, 1, 1))
    rps, ll, br, acc = _scores(_fit(tr, va, te), _y(te))
    rows.append({"scope": "series", "model": "Elo-only (LogReg on elo_diff)", "n": te.height,
                 "RPS": round(rps, 4), "logloss": round(ll, 4), "Brier": round(br, 4), "Accuracy": round(acc, 3)})
    # pooled World Cups (rolling 8y per Cup)
    ps, ys = [], []
    for wc in [2010, 2014, 2018, 2022]:
        start = mf.filter((pl.col("tournament") == "FIFA World Cup") & (pl.col("date").dt.year() == wc))["date"].min()
        win = mf.filter((pl.col("date") >= start.replace(year=start.year - 8)) & (pl.col("date") < start)).sort("date")
        k = int(win.height * 0.85)
        cup = mf.filter((pl.col("tournament") == "FIFA World Cup") & (pl.col("date").dt.year() == wc))
        if cup.is_empty() or k < 200:
            continue
        ps.append(_fit(win.head(k), win.slice(k), cup)); ys.append(_y(cup))
    p, y = np.vstack(ps), np.concatenate(ys)
    rps, ll, br, acc = _scores(p, y)
    rows.append({"scope": "worldcup", "model": "Elo-only (LogReg on elo_diff)", "n": len(y),
                 "RPS": round(rps, 4), "logloss": round(ll, 4), "Brier": round(br, 4), "Accuracy": round(acc, 3)})
    out = pl.DataFrame(rows)
    out.write_csv(config.FIGURES / "rev_b1_elo_only.csv")
    print(out)
    print("\nCompare to full-feature LogReg: series RPS 0.1685, WC RPS 0.2083 — the gap is the "
          "incremental value of the 107-feature table over Elo alone.")
    viz.save_table(viz.df_to_neurips_latex(out, label="tab:rev_elo_only", float_format="%.4f",
        caption="Elo-only reference (logistic on `elo_diff_eff` only): the dominant covariate without the "
                "rest of the feature table, for the series and pooled World Cups."), "rev_b1_elo_only")
    print("saved: rev_b1_elo_only.{csv,tex}")


if __name__ == "__main__":
    main()
