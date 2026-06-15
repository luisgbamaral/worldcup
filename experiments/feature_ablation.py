"""Feature-count ablation: is it better to use all ~120 features or a subset?

Walk-forward RPS as a function of the number of features kept (top-k by mutual
information, fit on each train fold only — leakage-safe). Tree models do implicit
selection so should be flat; linear / foundation models tend to prefer fewer,
cleaner inputs. TabPFN (cloud) is evaluated on the held-out slice for a few k.

    python experiments/feature_ablation.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldcup import config, modeling as M, viz  # noqa: E402

PRE_WC = dt.date(2026, 6, 10)
K_GRID = [10, 20, 30, 40, 60, 90]


def _data():
    mf = (pl.read_parquet(config.MATCH_FEATURES).sort("date", "match_id")
          .unique(subset="match_id", keep="first", maintain_order=True))
    return mf.filter((~pl.col("is_2026")) & (pl.col("date") <= PRE_WC))


def main():
    data = _data()
    cols = M.feature_columns(data)
    X, y, dates = M.to_numpy(data, cols), M.encode_result(data), data["date"].to_numpy()
    folds = M.walk_forward_folds(dates, n_folds=4)
    grid = [k for k in K_GRID if k < len(cols)] + [len(cols)]   # ... + "all"

    rows = []
    for name in ("LogReg", "CatBoost"):
        for k in grid:
            scores = []
            for tr, va in folds:
                idx = M.select_top_k(X[tr], y[tr], k)            # fit selection on train only
                pipe = M.build_classifier(name).fit(X[tr][:, idx], y[tr])
                scores.append(M.rps(M.proba_hda(pipe, X[va][:, idx]), y[va]))
            rows.append({"model": name, "k": k, "rps_mean": float(np.mean(scores)),
                         "rps_std": float(np.std(scores))})
            print(f"{name:9} k={k:3}  RPS={np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # TabPFN on the held-out slice only (cloud cost)
    if M.HAS_TABPFN:
        cut = sorted(set(dates.tolist()))[int(len(set(dates.tolist())) * 0.85)]
        tr, va = np.where(dates <= cut)[0], np.where(dates > cut)[0]
        for k in (20, 30, 60):
            idx = M.select_top_k(X[tr], y[tr], k)
            p = M.proba_hda(M.build_classifier("TabPFN").fit(X[tr][:, idx], y[tr]), X[va][:, idx])
            rows.append({"model": "TabPFN", "k": k, "rps_mean": M.rps(p, y[va]), "rps_std": 0.0})
            print(f"TabPFN    k={k:3}  RPS(test)={M.rps(p, y[va]):.4f}")

    res = pl.DataFrame(rows)
    res.write_parquet(config.PROCESSED / "feature_ablation.parquet")

    viz.set_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(viz.TEXTWIDTH_IN, viz.TEXTWIDTH_IN * 0.6))
    for i, name in enumerate(res["model"].unique(maintain_order=True)):
        d = res.filter(pl.col("model") == name).sort("k")
        ax.plot(d["k"], d["rps_mean"], label=name, **viz.style_for(i))
    ax.set(xlabel="number of features kept (top-k by mutual information)",
           ylabel="walk-forward RPS (lower is better)", title="Feature-count ablation")
    ax.legend()
    viz.save_fig(fig, "11_feature_ablation")

    best = res.sort("rps_mean").row(0, named=True)
    print(f"\nbest: {best['model']} with k={best['k']} (RPS {best['rps_mean']:.4f})")
    tex = viz.df_to_neurips_latex(
        res.with_columns(pl.col("rps_mean").round(4), pl.col("rps_std").round(4)),
        label="tab:feature_ablation", float_format="%.4f", bold_best="rps_mean",
        caption=("Feature-count ablation (walk-forward RPS, lower is better): performance vs. the "
                 "number of top-k mutual-information features. Trees are flat; selection helps the "
                 "linear and foundation models."))
    viz.save_table(tex, "feature_ablation")


if __name__ == "__main__":
    main()
