"""Revision D4 — stability of the greedy forward feature ranking.

Re-runs the greedy forward selection over >=10 random subsamples (different seeds) of the
full pre-2022 history and reports the stability of the top features: mean entry rank ±
spread, and how often each feature lands in the top-k. Guards against the headline ranking
(Elo first, is_knockout 7th, ...) being a single-seed artefact.

    python experiments/rev_d4_forward_stability.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from worldcup import config, features, modeling as M, viz  # noqa: E402
import feature_forward as FF  # noqa: E402

N_SEEDS = 12
MAX_STEPS = 20
SUB = 10000


def main():
    mf = (features.build_match_features_cached(dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026")))
    tr = mf.filter(pl.col("date") < dt.date(2022, 1, 1))
    va = mf.filter(pl.col("date").dt.year() == 2022)
    cols = M.feature_columns(tr)
    Xtr, Xva, _ = FF._imputed(tr, va, va, cols)
    ytr, yva = FF._y(tr), FF._y(va)

    ranks = {c: [] for c in cols}                  # entry rank per feature per seed (MAX_STEPS+1 if absent)
    for s in range(N_SEEDS):
        idx = np.random.RandomState(s).choice(len(ytr), min(SUB, len(ytr)), replace=False)
        order = FF.forward_order(Xtr[idx], ytr[idx], Xva, yva, cols, max_steps=MAX_STEPS)
        names = [cols[i] for i in order]
        for c in cols:
            ranks[c].append(names.index(c) + 1 if c in names else MAX_STEPS + 1)
        print(f"  seed {s:2}: {names[:8]}", flush=True)

    rows = []
    for c, rk in ranks.items():
        a = np.array(rk)
        rows.append({"feature": c, "mean_rank": round(float(a.mean()), 2),
                     "std_rank": round(float(a.std()), 2),
                     "top5_freq": round(float((a <= 5).mean()), 2),
                     "top10_freq": round(float((a <= 10).mean()), 2)})
    out = pl.DataFrame(rows).sort("mean_rank")
    out.write_csv(config.FIGURES / "rev_d4_forward_stability.csv")
    top = out.head(15)
    print(f"\n=== forward-ranking stability over {N_SEEDS} seeds (top 15 by mean entry rank) ===")
    with pl.Config(tbl_rows=15):
        print(top)
    viz.save_table(viz.df_to_neurips_latex(top, label="tab:rev_fwd_stability", float_format="%.2f",
        caption=(f"Stability of the greedy forward ranking over {N_SEEDS} random subsamples: mean entry "
                 "rank (\\textpm std) and the frequency of entering the top-5 / top-10.")),
        "rev_d4_forward_stability")
    print("\nsaved: rev_d4_forward_stability.{csv,tex}")


if __name__ == "__main__":
    main()
