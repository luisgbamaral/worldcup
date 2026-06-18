"""Revision A1 — multiplicity correction across all survival-test cells.

The survival analysis reports one one-sided bootstrap p per (scope, features, window) cell
(24 cells). Those were never corrected for multiple comparisons. Here we apply
Benjamini-Hochberg FDR and Holm jointly across all cells and flag which survive.

Bootstrap p can be exactly 0 (no resample crossed 0); we floor it at 1/(B+1) with B=2000
before correction, the smallest value the 2000-resample bootstrap can resolve.

    python experiments/rev_a1_multiplicity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldcup import config, viz  # noqa: E402

B = 2000
PFLOOR = 1.0 / (B + 1)


def main():
    s = pl.read_parquet(config.PROCESSED / "ablation_survival.parquet")
    s = s.with_columns(pl.col("p_fm_better").clip(PFLOOR, 1.0).alias("p_raw"))
    p = s["p_raw"].to_numpy()
    q = multipletests(p, method="fdr_bh")[1]
    holm = multipletests(p, method="holm")[1]
    s = s.with_columns(FDR_q=pl.Series(q).round(4), Holm_p=pl.Series(holm).round(4)) \
         .with_columns(survives_fdr=pl.col("FDR_q") < 0.05,
                       survives_holm=pl.col("Holm_p") < 0.05)
    order = ["1mo", "6mo", "1y", "2y", "5y", "8y"]
    s = s.with_columns(pl.col("window").cast(pl.Enum(order))).sort("scope", "features", "window")
    s.write_parquet(config.PROCESSED / "ablation_survival.parquet")
    s.write_csv(config.FIGURES / "ablation_survival.csv")

    show = s.select("scope", "features", "window", "best_fm", "best_classical",
                    "delta_fm_minus_cl", "p_raw", "FDR_q", "Holm_p", "survives_fdr")
    tex = viz.df_to_neurips_latex(
        show, label="tab:ablation_survival", float_format="%.4f", lower_is_better=False,
        caption=("Survival of the foundation-model advantage with multiplicity correction across all 24 "
                 "cells. $\\Delta$ = best tuned classical $-$ best FM per-match RPS (positive favours the "
                 "FM); p\\_raw is the one-sided clustered-bootstrap p (floored at $1/2001$); FDR\\_q is "
                 "Benjamini-Hochberg, Holm\\_p is Holm. survives\\_fdr at q$<$0.05."))
    viz.save_table(tex, "ablation_survival")

    with pl.Config(tbl_rows=30, tbl_cols=12, fmt_str_lengths=12):
        print(show)
    print(f"\nsignificant raw (p<0.05): {s.filter(pl.col('p_raw') < 0.05).height}/24")
    print(f"survive FDR (q<0.05):     {s.filter(pl.col('survives_fdr')).height}/24")
    print(f"survive Holm (p<0.05):    {s.filter(pl.col('survives_holm')).height}/24")
    print("\n--- cells surviving FDR ---")
    surv = s.filter(pl.col("survives_fdr")).select("scope", "features", "window",
                                                   "delta_fm_minus_cl", "FDR_q")
    print(surv if surv.height else "(none)")


if __name__ == "__main__":
    main()
