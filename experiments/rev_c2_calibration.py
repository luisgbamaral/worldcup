"""Revision C2 — multiclass calibration: Dirichlet vs per-class isotonic.

Per-class isotonic + renormalisation can distort the probability simplex. We add Dirichlet
calibration (matrix scaling on log-probabilities; Kull et al. 2019) as an alternative, fit
on the same temporal slice (2022), and report **class-conditional ECE** (home/draw/away) for
raw / isotonic / Dirichlet on the 2023+ test, plus reliability diagrams for the main models.
States whether the calibrator choice changes any conclusion.

    python experiments/rev_c2_calibration.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldcup import config, features, modeling as M, viz  # noqa: E402

_y = lambda d: np.where(d["home_score"].to_numpy() > d["away_score"].to_numpy(), 0,
                        np.where(d["home_score"].to_numpy() == d["away_score"].to_numpy(), 1, 2))


class Dirichlet:                                   # matrix scaling on log-probabilities
    def fit(self, p, y):
        self.clf = LogisticRegression(max_iter=2000, C=1.0).fit(np.log(np.clip(p, 1e-6, 1)), y)
        return self

    def predict(self, p):
        out = self.clf.predict_proba(np.log(np.clip(p, 1e-6, 1)))
        cls = list(self.clf.classes_)
        return out[:, [cls.index(i) for i in range(3)]]


def _cc_ece(p, y, bins=10):
    """class-conditional ECE: mean over classes of binned |confidence - frequency|."""
    edges = np.linspace(0, 1, bins + 1)
    per = []
    for k in range(3):
        conf, hit = p[:, k], (y == k).astype(float)
        e = 0.0
        for b in range(bins):
            m = (conf >= edges[b]) & (conf < edges[b + 1] if b < bins - 1 else conf <= 1.0)
            if m.sum():
                e += m.mean() * abs(conf[m].mean() - hit[m].mean())
        per.append(e)
    return per, float(np.mean(per))


def main():
    mf = (features.build_match_features_cached(dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026")))
    p = config.PROCESSED / "optimal_features.json"
    cols = json.loads(p.read_text("utf-8"))["features"] if p.exists() else M.feature_columns(mf)
    tr = mf.filter(pl.col("date") < dt.date(2022, 1, 1)).filter(pl.col("home_score").is_not_null()).sort("date")
    va = mf.filter(pl.col("date").dt.year() == 2022)
    te = mf.filter(pl.col("date") >= dt.date(2023, 1, 1))
    med = np.nan_to_num(np.nanmedian(M.to_numpy(tr, cols), axis=0), nan=0.0)
    fill = lambda d: np.where(np.isnan(M.to_numpy(d, cols)), med, M.to_numpy(d, cols))
    Xtr, Xva, Xte = fill(tr), fill(va), fill(te)
    ytr, yva, yte = _y(tr), _y(va), _y(te)

    rows, diagrams = [], {}
    for name in [m for m in ("LogReg", "CatBoost", "TabPFN", "TabICL") if m in M.available_classifiers()]:
        Xf, yf = (Xtr, ytr)
        if name in ("TabPFN", "TabICL") and len(ytr) > 10000:
            sub = np.random.RandomState(M.SEED).choice(len(ytr), 10000, replace=False); Xf, yf = Xtr[sub], ytr[sub]
        pipe = M.build_classifier(name).fit(Xf, yf)
        raw_va, raw_te = M.proba_hda(pipe, Xva), M.proba_hda(pipe, Xte)
        iso = M.apply_calibrators(raw_te, M.fit_calibrators(raw_va, yva))
        dir_te = Dirichlet().fit(raw_va, yva).predict(raw_te)
        for tag, pte in [("raw", raw_te), ("isotonic", iso), ("dirichlet", dir_te)]:
            per, m = _cc_ece(pte, yte)
            rows.append({"model": name, "calibrator": tag, "ECE_home": round(per[0], 4),
                         "ECE_draw": round(per[1], 4), "ECE_away": round(per[2], 4),
                         "ECE_mean": round(m, 4), "RPS": round(float(M.rps(pte, yte)), 4)})
        diagrams[name] = {"isotonic": iso, "dirichlet": dir_te}
        print(f"  {name}: done", flush=True)
    out = pl.DataFrame(rows)
    out.write_csv(config.FIGURES / "rev_c2_calibration.csv")
    with pl.Config(tbl_rows=40):
        print(out)
    viz.save_table(viz.df_to_neurips_latex(out, label="tab:rev_calibration", float_format="%.4f",
        caption=("Class-conditional ECE (home/draw/away) and RPS under raw / per-class isotonic / Dirichlet "
                 "calibration, test 2023+. Lower ECE is better.")), "rev_c2_calibration")

    # reliability diagram (draw class — the hard one) for the main models, isotonic vs Dirichlet
    viz.set_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(viz.TEXTWIDTH_IN, viz.TEXTWIDTH_IN * 0.45))
    edges = np.linspace(0, 1, 11); mids = (edges[:-1] + edges[1:]) / 2
    for ax, cal in zip(axes, ("isotonic", "dirichlet")):
        ax.plot([0, 1], [0, 1], "k:", lw=0.6)
        for i, name in enumerate(diagrams):
            p1 = diagrams[name][cal][:, 1]; hit = (yte == 1).astype(float)
            curve = [hit[(p1 >= edges[b]) & (p1 < edges[b + 1])].mean()
                     if ((p1 >= edges[b]) & (p1 < edges[b + 1])).sum() else np.nan for b in range(10)]
            ax.plot(mids, curve, marker="o", ms=2, label=name, **{k: v for k, v in
                    viz.style_for(i).items() if k != "marker"})
        ax.set(title=f"{cal} — draw", xlabel="predicted", ylabel="empirical", xlim=(0, .6), ylim=(0, .6))
    axes[1].legend(fontsize=5)
    viz.save_fig(fig, "18_reliability_draw")
    print("\nsaved: rev_c2_calibration.{csv,tex}, 18_reliability_draw.{pdf,png}")


if __name__ == "__main__":
    main()
