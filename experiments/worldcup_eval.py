"""One-step (per-match 1X2) evaluation — TWO scopes:

* **whole series**: predict every international match in a held-out period
  (train < 2022, validate 2022, test 2023+).
* **World Cups**: rolling 8-year-train per Cup (2010-2022), predict that Cup's matches.

Both use the **forward-selected optimal feature set** (`feature_forward.py`), the model
set LogReg / CatBoost / XGBoost + tune-free TabPFN / TabICL (Elo is a covariate, not a
baseline), isotonic calibration, and report RPS (primary), hits, accuracy, ECE.
Significance vs the simplest feature model **LogReg**: World-Cup-clustered (or yearly)
bootstrap + Diebold-Mariano (RPS) and McNemar (correctness), Holm-adjusted.

The tournament-level **multi-step** test (World Cup only) lives in ``worldcup_multistep.py``.

    python experiments/worldcup_eval.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldcup import config, features, modeling as M, viz  # noqa: E402

WORLD_CUPS = [2010, 2014, 2018, 2022]
BASELINE = "LogReg"
CHEAP = ["LogReg", "CatBoost", "XGBoost"]
SEED = M.SEED
_CLS = {"H": 0, "D": 1, "A": 2}
_y = lambda d: np.array([_CLS[r] for r in d["result"]])


def _optimal_cols(mf):
    p = config.PROCESSED / "optimal_features.json"
    if p.exists():
        return json.loads(p.read_text("utf-8"))["features"]
    print("[warn] optimal_features.json missing — using all features")
    return M.feature_columns(mf)


def _split(df, start, end):
    win = df.filter((pl.col("date") >= start) & (pl.col("date") < end)).sort("date")
    k = max(1, int(win.height * 0.85))
    return win.head(k), (win.slice(k) if win.height - k else win.tail(max(1, win.height // 7)))


def _predict(name, tr, va, te, cols):
    Xtr, ytr = M.to_numpy(tr, cols), _y(tr)
    sub = np.random.RandomState(SEED).choice(len(ytr), min(8000, len(ytr)), replace=False)
    pipe = M.build_classifier(name).fit(Xtr[sub], ytr[sub])
    cals = M.fit_calibrators(M.proba_hda(pipe, M.to_numpy(va, cols)), _y(va))
    return M.apply_calibrators(M.proba_hda(pipe, M.to_numpy(te, cols)), cals)


def _emit(model, group, te, p):
    y = _y(te)
    pred = np.argmax(p, axis=1)
    return [{"model": model, "group": group, "date": te["date"][i], "match_id": te["match_id"][i],
             "p_home": p[i, 0], "p_draw": p[i, 1], "p_away": p[i, 2], "y": int(y[i]),
             "correct": int(pred[i] == y[i]), "rps": M.rps(p[i:i + 1], y[i:i + 1])}
            for i in range(te.height)]


def _models():
    return CHEAP + [m for m in ("TabPFN", "TabICL", "TabDPT") if m in M.available_classifiers()]


def onestep_series(mf, cols):
    tr0 = mf.filter(pl.col("date") < dt.date(2022, 1, 1))
    va = mf.filter(pl.col("date").dt.year() == 2022)
    te = mf.filter(pl.col("date") >= dt.date(2023, 1, 1))
    rows = []
    for name in _models():
        try:
            rows += _emit(name, "series", te, _predict(name, tr0, va, te, cols))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] series {name}: {type(exc).__name__} {str(exc)[:70]}")
    # tag each row with its year for the clustered bootstrap / chart segments
    return pl.DataFrame(rows).with_columns(pl.col("date").dt.year().alias("group"))


def _wc_start(mf, wc):
    return mf.filter((pl.col("tournament") == "FIFA World Cup")
                     & (pl.col("date").dt.year() == wc))["date"].min()


def onestep_worldcup(mf, cols):
    rows = []
    for wc in WORLD_CUPS:
        start = _wc_start(mf, wc)
        tr, va = _split(mf, start.replace(year=start.year - 8), start)
        te = mf.filter((pl.col("tournament") == "FIFA World Cup")
                       & (pl.col("date").dt.year() == wc)).sort("date")
        if te.is_empty() or tr.height < 200:
            continue
        for name in _models():
            try:
                rows += _emit(name, wc, te, _predict(name, tr, va, te, cols))
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] WC {name}@{wc}: {type(exc).__name__} {str(exc)[:60]}")
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# metrics, significance, chart (shared)
# --------------------------------------------------------------------------- #
def _summary(df):
    g = df.group_by("model").agg(n=pl.len(), RPS=pl.col("rps").mean(),
                                 Hits=pl.col("correct").sum(), Accuracy=pl.col("correct").mean())
    ece = {m[0]: M.ece(d.select("p_home", "p_draw", "p_away").to_numpy(), d["y"].to_numpy())
           for m, d in df.group_by("model")}
    return g.with_columns(ECE=pl.col("model").replace_strict(ece)).sort("RPS")


def _tests(df):
    base = df.filter(pl.col("model") == BASELINE).sort("match_id")
    rng = np.random.default_rng(SEED)
    rows = []
    for model in [m for m in df["model"].unique().to_list() if m != BASELINE]:
        mo = df.filter(pl.col("model") == model).sort("match_id")
        j = base.select("match_id", "group", rps_b="rps", c_b="correct").join(
            mo.select("match_id", rps_m="rps", c_m="correct"), on="match_id")
        d = (j["rps_b"] - j["rps_m"]).to_numpy()
        grp = j["group"].to_numpy(); uniq = np.unique(grp)
        boot = [np.concatenate([d[grp == rng.choice(uniq)] for _ in uniq]).mean() for _ in range(2000)]
        p_boot = float(np.mean(np.array(boot) <= 0))
        cb, cm = j["c_b"].to_numpy(), j["c_m"].to_numpy()
        tb = [[int(((cb == 1) & (cm == 1)).sum()), int(((cb == 1) & (cm == 0)).sum())],
              [int(((cb == 0) & (cm == 1)).sum()), int(((cb == 0) & (cm == 0)).sum())]]
        rows.append({"model": model, "dRPS": round(float(d.mean()), 4),
                     "CI95": f"[{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]",
                     "p_boot": round(p_boot, 4),
                     "p_McNemar": round(float(mcnemar(tb, exact=False, correction=True).pvalue), 4)})
    for r, ph in zip(rows, multipletests([r["p_boot"] for r in rows], method="holm")[1]):
        r["p_boot_holm"] = round(float(ph), 4)
    return pl.DataFrame(rows)


def cumulative_series(df):
    order = (df.filter(pl.col("model") == BASELINE).sort("date", "match_id")
             .with_row_index("idx").select("match_id", "idx", "date", "group"))
    out = order.clone()
    for model in sorted(df["model"].unique().to_list()):
        s = (df.filter(pl.col("model") == model).join(order, on="match_id")
             .sort("idx").with_columns(cum=pl.col("correct").cum_sum()))
        out = out.join(s.select("match_id", **{model: "cum"}), on="match_id", how="left")
    return out.sort("idx")


def _report(df, tag, boundaries_label):
    summ, tests = _summary(df), _tests(df)
    print(f"\n=== {tag}: per-model ===\n{summ}\n--- significance vs {BASELINE} ---\n{tests}")
    show = summ.join(tests, on="model", how="left").with_columns(
        pl.col("RPS").round(4), pl.col("Accuracy").round(3), pl.col("ECE").round(4)).sort("RPS")
    tex = viz.df_to_neurips_latex(
        show.select("model", "RPS", "Hits", "Accuracy", "ECE", "dRPS", "p_boot_holm", "p_McNemar"),
        label=f"tab:onestep_{tag}", float_format="%.4f", bold_best="RPS",
        caption=(f"One-step 1X2 forecasting — {boundaries_label}. RPS lower is better; "
                 "Hits/Accuracy higher. Optimal forward-selected features, Elo as covariate; "
                 f"Δ/p vs {BASELINE} (Holm bootstrap + McNemar). Foundation models tune-free."))
    viz.save_table(tex, f"onestep_{tag}")
    cum = cumulative_series(df)
    viz.set_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(viz.TEXTWIDTH_IN, viz.TEXTWIDTH_IN * 0.6))
    for i, model in enumerate(sorted(df["model"].unique().to_list())):
        ax.plot(cum["idx"], cum[model], label=model, **viz.style_for(i))
    for b in cum.group_by("group").agg(pl.col("idx").min())["idx"].sort():
        ax.axvline(b, color="0.85", lw=0.5, zorder=0)
    ax.set(xlabel=f"held-out matches (chronological; {boundaries_label})",
           ylabel="cumulative correct 1X2", title=f"Cumulative hits — {tag}")
    ax.legend(ncol=2, fontsize=6)
    viz.save_fig(fig, f"12_cumhits_{tag}")
    cum.write_csv(config.FIGURES / f"cumulative_hits_{tag}.csv")


def main():
    mf = (features.build_match_features(train_cut=dt.date(2002, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026")))
    cols = _optimal_cols(mf)
    print(f"optimal features ({len(cols)}): {cols}")
    series = onestep_series(mf, cols); series.write_parquet(config.PROCESSED / "onestep_series.parquet")
    _report(series, "series", "whole historical series, test 2023+")
    wc = onestep_worldcup(mf, cols); wc.write_parquet(config.PROCESSED / "onestep_worldcup.parquet")
    _report(wc, "worldcup", "World Cups 2010-2022, rolling 8-year train")
    print("\nsaved one-step (series + worldcup) tables, charts, parquets")


if __name__ == "__main__":
    main()
