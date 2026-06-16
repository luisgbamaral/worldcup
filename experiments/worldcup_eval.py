"""World-Cup hit-count evaluation (1X2): rolling 8-year-train per tournament.

For each World Cup in ``WORLD_CUPS`` we train on the 8 years immediately before its
first match (rolling origin) and predict every match of that tournament, freezing
``P(H/D/A)``. Models: three rating-only baselines (ELO-Classic / ELO-World /
Pi-Rating), LogReg / CatBoost / XGBoost classifiers on the (single-rating) feature
table with CFS selection, and the tune-free foundation models (TabPFN / TabICL /
TabDPT when available). Metrics: RPS (primary), hits, accuracy, ECE. Significance
vs the ELO-World baseline: bootstrap clustered by World Cup + Diebold-Mariano
(RPS) and McNemar (correctness), with Holm multiplicity control.

    python experiments/worldcup_eval.py [--sims-off-fms]

Outputs: reports/tables/worldcup_eval.tex, reports/figures/12_cumulative_hits.{pdf,png},
data/processed/worldcup_eval.parquet, reports/figures/cumulative_hits.csv.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldcup import config, data, features, modeling as M, ratings as R, viz  # noqa: E402
from worldcup.clean import canon_team  # noqa: E402

WORLD_CUPS = [2010, 2014, 2018, 2022]
BASELINE = "ELO-World"
RATING_MODELS = {"ELO-Classic": R.rate_classic, "ELO-World": R.rate_world, "Pi-Rating": R.rate_pi}
CLF_MODELS = ["LogReg", "CatBoost", "XGBoost"]
SEED = M.SEED
_CLS = {"H": 0, "D": 1, "A": 2}


def _y(df):
    return np.array([_CLS[r] for r in df["result"]]) if "result" in df.columns else \
        np.where(df["home_score"].to_numpy() > df["away_score"].to_numpy(), 0,
                 np.where(df["home_score"].to_numpy() == df["away_score"].to_numpy(), 1, 2))


def _wc_start(df, wc):
    return df.filter((pl.col("tournament") == "FIFA World Cup")
                     & (pl.col("date").dt.year() == wc))["date"].min()


def _split(df, start, end):
    """train window [start, end) split into fit (first 85%) + temporal val (last 15%)."""
    win = df.filter((pl.col("date") >= start) & (pl.col("date") < end)).sort("date")
    k = max(1, int(win.height * 0.85))
    return win.head(k), (win.slice(k) if win.height - k else win.tail(max(1, win.height // 7)))


def _rating_preds():
    played = data.load_results(played_only=True).with_columns(
        pl.when(pl.col("home_score") > pl.col("away_score")).then(pl.lit("H"))
          .when(pl.col("home_score") == pl.col("away_score")).then(pl.lit("D"))
          .otherwise(pl.lit("A")).alias("result"))
    rows = []
    for name, fn in RATING_MODELS.items():
        # canonical match_id (same hash as the feature table) so all models align
        rated = fn(played).with_columns(canon_team("home_team").alias("home_team"),
                                        canon_team("away_team").alias("away_team"))
        rated = rated.with_columns(features._match_id().alias("match_id"))
        for wc in WORLD_CUPS:
            start = _wc_start(rated, wc)
            tr, va = _split(rated, start.replace(year=start.year - 8), start)
            wcm = rated.filter((pl.col("tournament") == "FIFA World Cup")
                               & (pl.col("date").dt.year() == wc)).sort("date")
            p = R.Readout().fit(tr, va).predict(wcm)
            rows += _emit(name, wc, wcm, p)
    return rows


def _clf_fm_preds(mf):
    rows = []
    fms = [m for m in ("TabPFN", "TabICL", "TabDPT") if m in M.available_classifiers()]
    for wc in WORLD_CUPS:
        start = _wc_start(mf, wc)
        tr, va = _split(mf, start.replace(year=start.year - 8), start)
        wcm = mf.filter((pl.col("tournament") == "FIFA World Cup")
                        & (pl.col("date").dt.year() == wc)).sort("date")
        if wcm.is_empty() or tr.height < 200:
            continue
        cols = M.feature_columns(tr)
        sel = M.select_features_cfs(M.to_numpy(tr, cols), _y(tr), cols)
        Xtr, ytr = M.to_numpy(tr, sel), _y(tr)
        Xva, yva = M.to_numpy(va, sel), _y(va)
        Xwc = M.to_numpy(wcm, sel)
        sub = np.random.RandomState(SEED).choice(len(ytr), min(8000, len(ytr)), replace=False)
        for name in CLF_MODELS + fms:
            try:
                pipe = M.build_classifier(name).fit(Xtr[sub], ytr[sub])
                cals = M.fit_calibrators(M.proba_hda(pipe, Xva), yva)
                p = M.apply_calibrators(M.proba_hda(pipe, Xwc), cals)
                rows += _emit(name, wc, wcm, p)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {name} @ {wc}: {type(exc).__name__} {str(exc)[:80]}")
    return rows


def _emit(model, wc, wcm, p):
    y = _y(wcm)
    pred = np.argmax(p, axis=1)
    out = []
    for i, m in enumerate(wcm.iter_rows(named=True)):
        out.append({"model": model, "wc": wc, "date": m["date"], "match_id": m.get("match_id"),
                    "p_home": p[i, 0], "p_draw": p[i, 1], "p_away": p[i, 2],
                    "y": int(y[i]), "correct": int(pred[i] == y[i]),
                    "rps": M.rps(p[i:i + 1], y[i:i + 1])})
    return out


# --------------------------------------------------------------------------- #
# metrics + significance
# --------------------------------------------------------------------------- #
def _summary(df):
    g = df.group_by("model").agg(
        n=pl.len(), RPS=pl.col("rps").mean(), Hits=pl.col("correct").sum(),
        Accuracy=pl.col("correct").mean())
    ece = {m[0]: M.ece(d.select("p_home", "p_draw", "p_away").to_numpy(), d["y"].to_numpy())
           for m, d in df.group_by("model")}
    return g.with_columns(ECE=pl.col("model").replace_strict(ece)).sort("RPS")


def _tests(df):
    base = df.filter(pl.col("model") == BASELINE).sort("match_id")
    rows, pvals = [], []
    for model in [m for m in df["model"].unique().to_list() if m != BASELINE]:
        mo = df.filter(pl.col("model") == model).sort("match_id")
        j = base.select("match_id", "wc", rps_b="rps", c_b="correct").join(
            mo.select("match_id", rps_m="rps", c_m="correct"), on="match_id")
        d = (j["rps_b"] - j["rps_m"]).to_numpy()                 # >0 => model better (lower RPS)
        delta = float(d.mean())
        # block bootstrap clustered by World Cup
        wcs = j["wc"].to_numpy()
        uniq = np.unique(wcs)
        rng = np.random.default_rng(SEED)
        boot = [np.concatenate([d[wcs == rng.choice(uniq)] for _ in uniq]).mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_boot = float(np.mean(np.array(boot) <= 0))            # one-sided H1: delta>0
        dm_p = float(2 * (1 - _norm_cdf(abs(delta) / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12))))
        cb, cm = j["c_b"].to_numpy(), j["c_m"].to_numpy()
        tb = [[int(((cb == 1) & (cm == 1)).sum()), int(((cb == 1) & (cm == 0)).sum())],
              [int(((cb == 0) & (cm == 1)).sum()), int(((cb == 0) & (cm == 0)).sum())]]
        mc_p = float(mcnemar(tb, exact=False, correction=True).pvalue)
        pvals.append(p_boot)
        rows.append({"model": model, "dRPS": round(delta, 4), "CI95": f"[{lo:.4f}, {hi:.4f}]",
                     "p_boot": round(p_boot, 4), "p_DM": round(dm_p, 4), "p_McNemar": round(mc_p, 4)})
    holm = multipletests([r["p_boot"] for r in rows], method="holm")[1]
    for r, ph in zip(rows, holm):
        r["p_boot_holm"] = round(float(ph), 4)
    return pl.DataFrame(rows)


def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


# --------------------------------------------------------------------------- #
# PART F — cumulative correct-predictions chart
# --------------------------------------------------------------------------- #
def cumulative_series(df):
    """Per-model running sum of hits over the chronologically-ordered held-out matches."""
    order = (df.filter(pl.col("model") == BASELINE).sort("date", "match_id")
             .with_row_index("idx").select("match_id", "idx", "date", "wc"))
    out = order.clone()
    for model in sorted(df["model"].unique().to_list()):
        s = (df.filter(pl.col("model") == model).join(order, on="match_id")
             .sort("idx").with_columns(cum=pl.col("correct").cum_sum()))
        out = out.join(s.select("match_id", **{model: "cum"}), on="match_id", how="left")
    return out.sort("idx")


def _cumulative_chart(df):
    cum = cumulative_series(df)
    models = sorted(df["model"].unique().to_list())
    viz.set_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(viz.TEXTWIDTH_IN, viz.TEXTWIDTH_IN * 0.6))
    order = cum.select("idx", "wc")
    for i, model in enumerate(models):
        ax.plot(cum["idx"], cum[model], label=model, **viz.style_for(i))
    csv = cum
    for b in order.group_by("wc").agg(pl.col("idx").min())["idx"].sort():
        ax.axvline(b, color="0.85", lw=0.5, zorder=0)
    ax.set(xlabel="held-out World-Cup matches (chronological)",
           ylabel="cumulative correct 1X2 predictions", title="Cumulative hits across the last four World Cups")
    ax.legend(ncol=2, fontsize=6)
    viz.save_fig(fig, "12_cumulative_hits")
    csv.sort("idx").write_csv(config.FIGURES / "cumulative_hits.csv")


def main():
    print("building extended features (train_cut=2002)...")
    mf = (features.build_match_features(train_cut=dt.date(2002, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026")))
    print("rating baselines..."); rows = _rating_preds()
    print("classifiers + foundation models..."); rows += _clf_fm_preds(mf)
    df = pl.DataFrame(rows)
    df.write_parquet(config.PROCESSED / "worldcup_eval.parquet")

    summ = _summary(df)
    tests = _tests(df)
    print("\n=== per-model (pooled over", len(WORLD_CUPS), "World Cups) ===")
    print(summ)
    print("\n=== significance vs", BASELINE, "===")
    print(tests)

    show = summ.join(tests, on="model", how="left").with_columns(
        pl.col("RPS").round(4), pl.col("Accuracy").round(3), pl.col("ECE").round(4)).sort("RPS")
    tex = viz.df_to_neurips_latex(
        show.select("model", "RPS", "Hits", "Accuracy", "ECE", "dRPS", "p_boot_holm", "p_McNemar"),
        label="tab:worldcup_eval", float_format="%.4f", bold_best="RPS",
        caption=("1X2 forecasting on the last four World Cups (rolling 8-year train per "
                 f"tournament; pooled). RPS lower is better; Hits/Accuracy higher. ΔRPS and "
                 f"p-values are vs the {BASELINE} baseline (Holm-adjusted bootstrap; McNemar on "
                 "correctness). Foundation models are tune-free."))
    viz.save_table(tex, "worldcup_eval")
    _cumulative_chart(df)
    print("\nsaved: worldcup_eval.{parquet,tex}, 12_cumulative_hits.{pdf,png}, cumulative_hits.csv")


if __name__ == "__main__":
    main()
