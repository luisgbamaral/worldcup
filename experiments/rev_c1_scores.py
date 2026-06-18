"""Revision C1 — complementary proper scores (log-loss + Brier) on the main tables.

Adds multiclass log-loss and multiclass Brier next to RPS for the one-step series and
World-Cup tables (computed from the saved per-match probabilities), and reports whether the
model ranking is robust across the three proper scoring rules. RPS stays primary.

    python experiments/rev_c1_scores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldcup import config, viz  # noqa: E402


def _table(name):
    d = pl.read_parquet(config.PROCESSED / f"{name}.parquet")
    rows = []
    for m, g in d.group_by("model"):
        p = g.select("p_home", "p_draw", "p_away").to_numpy()
        y = g["y"].to_numpy()
        Y = np.eye(3)[y]
        rps = float(g["rps"].mean())
        ll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
        brier = float(((p - Y) ** 2).sum(1).mean())
        acc = float((p.argmax(1) == y).mean())
        rows.append({"model": m[0], "n": g.height, "RPS": round(rps, 4),
                     "logloss": round(ll, 4), "Brier": round(brier, 4), "Accuracy": round(acc, 3)})
    return pl.DataFrame(rows).sort("RPS")


def _rank_agreement(t):
    r = {c: t.sort(c)["model"].to_list() for c in ("RPS", "logloss", "Brier")}
    same = r["RPS"] == r["logloss"] == r["Brier"]
    return same, r


def main():
    for name, label in [("onestep_series", "whole series, test 2023+"),
                        ("onestep_worldcup", "World Cups 2010-2022")]:
        t = _table(name)
        same, r = _rank_agreement(t)
        print(f"\n=== {name} ({label}) ===")
        print(t)
        print(f"ranking identical across RPS/logloss/Brier: {same}")
        if not same:
            for c in ("RPS", "logloss", "Brier"):
                print(f"  {c:8}: {r[c]}")
        t.write_csv(config.FIGURES / f"rev_c1_{name}.csv")
        viz.save_table(viz.df_to_neurips_latex(
            t, label=f"tab:rev_c1_{name}", float_format="%.4f", bold_best="RPS",
            caption=(f"One-step {label}: RPS (primary), multiclass log-loss and Brier, with accuracy. "
                     "Lower is better for the three proper scores.")), f"rev_c1_{name}")
    print("\nsaved: rev_c1_{onestep_series,onestep_worldcup}.{csv,tex}")


if __name__ == "__main__":
    main()
