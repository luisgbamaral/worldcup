"""Multi-step tournament evaluation (World Cup): simulate each Cup to the champion.

For the last five World Cups (32-team format), train each model on the **5 years
before** the tournament (frozen — no in-tournament updates) on the **optimal feature
set** from ``feature_forward.py``. Then forward-simulate the whole tournament:
predict the group matches -> standings (points; Elo tiebreak, since goals are out of
scope) -> top 2 advance -> the standard 32-team knockout bracket -> champion / runner-up
/ third. Monte-Carlo gives each team's P(champion / runner-up / third); the predicted
podium is the arg-max of each. Score: +1 per correctly-placed podium position (max 3
per Cup), accumulated across the five Cups.

This is the **multi-step** test (tournament-level, World Cup). The single-step,
whole-series per-match evaluation lives in ``worldcup_eval.py``.

    python experiments/worldcup_multistep.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldcup import config, elo as E, features, modeling as M, viz  # noqa: E402
from worldcup.clean import canon_team  # noqa: E402

WORLD_CUPS = [2006, 2010, 2014, 2018, 2022]
N_SIMS = 10000
MODELS = ["LogReg", "CatBoost", "XGBoost"]
PODIUM = {  # (champion, runner-up, third) — historical
    2006: ("Italy", "France", "Germany"), 2010: ("Spain", "Netherlands", "Germany"),
    2014: ("Germany", "Argentina", "Netherlands"), 2018: ("France", "Croatia", "Belgium"),
    2022: ("Argentina", "France", "Croatia")}
# standard 32-team bracket: R16 slots, then a tree of winner indices
R16 = [("1A", "2B"), ("1C", "2D"), ("1E", "2F"), ("1G", "2H"),
       ("1B", "2A"), ("1D", "2C"), ("1F", "2E"), ("1H", "2G")]
QF = [(0, 1), (2, 3), (4, 5), (6, 7)]
SF = [(0, 1), (2, 3)]


def _groups(year):
    doc = json.loads((config.WORLDCUP_JSON_DIR / str(year) / "worldcup.json").read_text("utf-8"))
    groups, fixtures = {}, {}
    for m in doc["matches"]:
        g = m.get("group")
        if not g:
            continue
        letter = g.split()[-1]
        h, a = canon_team_str(m["team1"]), canon_team_str(m["team2"])
        groups.setdefault(letter, set()).update([h, a])
        fixtures.setdefault(letter, []).append((h, a))
    return {k: sorted(v) for k, v in groups.items()}, fixtures


def canon_team_str(name):
    return pl.DataFrame({"t": [name]}).select(canon_team("t"))[0, 0]


# --------------------------------------------------------------------------- #
# feature construction for arbitrary pairings (from per-team snapshots)
# --------------------------------------------------------------------------- #
def _snapshots(mf, teams, before):
    """Latest pre-tournament per-team feature values (from the match feature table)."""
    snap = {}
    for t in teams:
        rows = mf.filter(((pl.col("home_team") == t) | (pl.col("away_team") == t))
                         & (pl.col("date") < before)).sort("date")
        if rows.is_empty():
            continue
        r = rows.row(-1, named=True)
        side = "home" if r["home_team"] == t else "away"
        d = {b: r.get(f"{b}_{side}") for b in features.TEAM_FEATS}
        d["elo"] = r["home_elo_pre"] if side == "home" else r["away_elo_pre"]
        snap[t] = d
    return snap


def _pair_row(si, sj, cols, year, med):
    out = []
    for c in cols:
        if c == "neutral":
            v = 1.0
        elif c == "year":
            v = float(year)
        elif c == "is_world_cup" or c == "is_competitive":
            v = 1.0
        elif c == "tournament_importance":
            v = 60.0
        elif c in ("home_elo_pre",):
            v = si["elo"]
        elif c in ("away_elo_pre",):
            v = sj["elo"]
        elif c in ("elo_diff", "elo_diff_eff"):
            v = si["elo"] - sj["elo"]
        elif c == "elo_sum":
            v = si["elo"] + sj["elo"]
        elif c == "elo_winprob":
            v = 1 / (1 + 10 ** (-(si["elo"] - sj["elo"]) / 400))
        elif c.endswith("_diff") and c[:-5] in features.TEAM_FEATS:
            v = (si.get(c[:-5]) or 0) - (sj.get(c[:-5]) or 0)
        elif c.endswith("_home") and c[:-5] in features.TEAM_FEATS:
            v = si.get(c[:-5])
        elif c.endswith("_away") and c[:-5] in features.TEAM_FEATS:
            v = sj.get(c[:-5])
        else:
            v = med.get(c)
        out.append(med.get(c) if v is None else v)
    return out


def _pair_matrix(model_pipe, snap, teams, cols, year, med):
    """P(H/D/A) for every ordered pairing (i home vs j away)."""
    idx, X = [], []
    present = [t for t in teams if t in snap]
    for i in present:
        for j in present:
            if i != j:
                idx.append((i, j)); X.append(_pair_row(snap[i], snap[j], cols, year, med))
    p = M.proba_hda(model_pipe, np.array(X, float))
    P = {}
    for (i, j), pr in zip(idx, p):
        P.setdefault(i, {})[j] = pr
    return P


# --------------------------------------------------------------------------- #
# one Monte-Carlo tournament
# --------------------------------------------------------------------------- #
def _simulate(P, groups, fixtures, elo, rng):
    pts = {t: 0 for g in groups.values() for t in g}
    for letter, fx in fixtures.items():
        for h, a in fx:
            if h not in P or a not in P.get(h, {}):
                continue
            ph, pd, pa = P[h][a]
            r = rng.random()
            if r < ph:
                pts[h] += 3
            elif r < ph + pd:
                pts[h] += 1; pts[a] += 1
            else:
                pts[a] += 3
    pos = {}
    for letter, teams in groups.items():
        ranked = sorted(teams, key=lambda t: (pts.get(t, 0), elo.get(t, 1500)), reverse=True)
        pos[f"1{letter}"], pos[f"2{letter}"] = ranked[0], ranked[1]

    def winner(a, b):
        if a not in P or b not in P.get(a, {}):
            return a if elo.get(a, 1500) >= elo.get(b, 1500) else b
        ph, pd, pa = P[a][b]
        return a if rng.random() < (ph + 0.5 * pd) else b

    r16w = [winner(pos[s1], pos[s2]) for s1, s2 in R16]
    qfw = [winner(r16w[i], r16w[j]) for i, j in QF]
    sf = [(qfw[i], qfw[j]) for i, j in SF]
    sfw = [winner(a, b) for a, b in sf]
    sfl = [b if w == a else a for (a, b), w in zip(sf, sfw)]
    champ = winner(sfw[0], sfw[1])
    runner = sfw[1] if champ == sfw[0] else sfw[0]
    third = winner(sfl[0], sfl[1])
    return champ, runner, third


def _eval_wc(year, mf, elo, cols, med):
    groups, fixtures = _groups(year)
    teams = [t for g in groups.values() for t in g]
    start = mf.filter((pl.col("tournament") == "FIFA World Cup")
                      & (pl.col("date").dt.year() == year))["date"].min()
    train = mf.filter((pl.col("date") >= start.replace(year=start.year - 5)) & (pl.col("date") < start))
    snap = _snapshots(mf, teams, start)
    ytr = np.where(train["home_score"].to_numpy() > train["away_score"].to_numpy(), 0,
                   np.where(train["home_score"].to_numpy() == train["away_score"].to_numpy(), 1, 2))
    Xtr = np.nan_to_num(M.to_numpy(train, cols))
    fms = [m for m in ("TabPFN", "TabICL", "TabDPT") if m in M.available_classifiers()]
    rows = []
    for name in MODELS + fms:
        try:
            pipe = M.build_classifier(name).fit(Xtr, ytr)
            P = _pair_matrix(pipe, snap, teams, cols, year, med)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {name} @ {year}: {type(exc).__name__} {str(exc)[:70]}"); continue
        rng = np.random.default_rng(M.SEED)
        c = {"champ": {}, "runner": {}, "third": {}}
        for _ in range(N_SIMS):
            ch, ru, th = _simulate(P, groups, fixtures, elo, rng)
            for key, t in (("champ", ch), ("runner", ru), ("third", th)):
                c[key][t] = c[key].get(t, 0) + 1
        amax = lambda d, ex=(): max((t for t in d if t not in ex), key=lambda t: d[t], default=None)
        p1 = amax(c["champ"]); p2 = amax(c["runner"], (p1,)); p3 = amax(c["third"], (p1, p2))
        act = PODIUM[year]
        pts = int(p1 == act[0]) + int(p2 == act[1]) + int(p3 == act[2])
        rows.append({"model": name, "wc": year, "pred_champion": p1, "actual_champion": act[0],
                     "pred_runner": p2, "pred_third": p3, "podium_points": pts,
                     "p_champion": round(c["champ"].get(p1, 0) / N_SIMS, 3)})
        print(f"  {name:9} {year}: champ {p1} (real {act[0]}) | podium pts {pts}/3")
    return rows


def main():
    opt = json.loads((config.PROCESSED / "optimal_features.json").read_text("utf-8"))
    cols = opt["features"]
    print(f"optimal feature set ({len(cols)}): {cols}")
    mf = (features.build_match_features_cached(train_cut=dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026")))
    med = {c: float(np.nan_to_num(np.nanmedian(M.to_numpy(mf, [c])))) for c in cols}
    elo = {r["team"]: r["rating"] for r in E.latest_ratings()
           .with_columns(canon_team("team").alias("team")).select("team", "rating").iter_rows(named=True)}

    rows = []
    for wc in WORLD_CUPS:
        rows += _eval_wc(wc, mf, elo, cols, med)
    df = pl.DataFrame(rows)
    df.write_parquet(config.PROCESSED / "worldcup_multistep.parquet")

    tot = df.group_by("model").agg(podium_points=pl.col("podium_points").sum(),
                                   champions_hit=(pl.col("pred_champion") == pl.col("actual_champion")).sum()
                                   ).sort("podium_points", descending=True)
    print("\n=== total (5 World Cups) ===\n", tot)
    tex = viz.df_to_neurips_latex(
        df.select("model", "wc", "pred_champion", "actual_champion", "pred_runner", "pred_third",
                  "podium_points").sort("wc", "model"),
        label="tab:worldcup_multistep", float_format="%.0f", bold_best="podium_points",
        lower_is_better=False,
        caption=("Multi-step World-Cup forecast: predicted vs actual champion, predicted podium, and "
                 "podium points (+1 per correctly-placed position, max 3/Cup) for the last five Cups. "
                 "Models trained on the 5 years before each Cup, frozen, on the forward-selected "
                 "optimal feature set; the tournament is simulated forward to the champion."))
    viz.save_table(tex, "worldcup_multistep")
    _cumulative_chart(df)
    print("saved: worldcup_multistep.{parquet,tex}, 14_multistep_cumpoints.{pdf,png}, multistep_points.csv")


def _cumulative_chart(df):
    viz.set_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(viz.TEXTWIDTH_IN, viz.TEXTWIDTH_IN * 0.6))
    wcs = WORLD_CUPS
    csv = pl.DataFrame({"wc": wcs})
    for i, name in enumerate(sorted(df["model"].unique().to_list())):
        s = df.filter(pl.col("model") == name).sort("wc")
        cum = np.cumsum([s.filter(pl.col("wc") == w)["podium_points"].sum() for w in wcs])
        ax.plot(range(len(wcs)), cum, label=name, marker="o", **{k: v for k, v in
                viz.style_for(i).items() if k != "marker"})
        csv = csv.with_columns(pl.Series(name, cum))
    for x in range(len(wcs)):
        ax.axvline(x, color="0.85", lw=0.5, zorder=0)
    ax.set_xticks(range(len(wcs))); ax.set_xticklabels(wcs)
    ax.set(xlabel="World Cup", ylabel="cumulative podium points",
           title="Multi-step: cumulative podium points by model")
    ax.legend(ncol=2, fontsize=6)
    viz.save_fig(fig, "14_multistep_cumpoints")
    csv.write_csv(config.FIGURES / "multistep_points.csv")


if __name__ == "__main__":
    main()
