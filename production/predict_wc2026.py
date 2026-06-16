"""Production forecast: who wins the 2026 World Cup (48-team format).

Self-contained -- depends only on the ``worldcup`` engine in ``src/`` (features, Elo,
modeling), NOT on the paper/experiments (which can be deleted after the Cup). It:

1. reads the official group schedule (``data/raw/fixtures/wc2026_group_stage_schedule.csv``),
2. maps each venue to its host country so home advantage applies ONLY when a host plays
   in its own country (USA in the USA, Mexico in Mexico, Canada in Canada),
3. snapshots every team's latest pre-tournament feature vector and trains the chosen
   classifier (default TabPFN; falls back to CatBoost/LogReg) on the matches before kickoff,
4. simulates the tournament 10,000 times -- group stage -> 12 winners + 12 runners-up + 8
   best thirds -> a strength-reseeded 32-team knockout bracket -> champion,
5. reports each team's P(champion / final / semifinal), for two modes:
   * **pre-tournament** (ignore results so far) and
   * **conditional** (use the group games already played, simulate the rest).

    python production/predict_wc2026.py
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldcup import config, elo as E, features, modeling as M  # noqa: E402
from worldcup.clean import canon_team  # noqa: E402

FIXTURES = ROOT / "data" / "raw" / "fixtures" / "wc2026_group_stage_schedule.csv"
OUT = Path(__file__).resolve().parent / "outputs"
KICKOFF = dt.date(2026, 6, 11)
TRAIN_YEARS = 8
N_SIMS = 10000
HOME_ADV = 65.0                      # Elo home bump, only for a host in its own country
MODEL_PRIORITY = [os.environ.get("WC2026_MODEL", "TabPFN"), "CatBoost", "LogReg"]
SEED = 42
_CLS = {0: "H", 1: "D", 2: "A"}


# --------------------------------------------------------------------------- #
# fixtures + host mapping
# --------------------------------------------------------------------------- #
def _venue_country(v: str) -> str:
    """Host country as the canonical team name (so it matches canon_team output)."""
    s = (v or "").lower()
    if any(k in s for k in ("mexico", "zapopan", "guadalupe")):
        return "Mexico"
    if any(k in s for k in ("toronto", "vancouver")):
        return "Canada"
    return "USA"                            # every other listed venue is in the USA


def _canon(name: str) -> str:
    return pl.DataFrame({"t": [name]}).select(canon_team("t"))[0, 0]


def load_fixtures():
    df = pl.read_csv(FIXTURES)
    rows = []
    for r in df.iter_rows(named=True):
        h, a = _canon(r["home_team"]), _canon(r["away_team"])
        country = _venue_country(r["venue"])
        host = h if h == country else (a if a == country else None)   # host playing at home
        played = r["home_score"] is not None and r["away_score"] is not None
        res = None
        if played:
            res = 0 if r["home_score"] > r["away_score"] else (1 if r["home_score"] == r["away_score"] else 2)
        rows.append({"group": r["group"], "home": h, "away": a, "host": host,
                     "played": played, "result": res})
    return pl.DataFrame(rows)


def groups_of(fx):
    g = {}
    for r in fx.iter_rows(named=True):
        g.setdefault(r["group"], set()).update([r["home"], r["away"]])
    return {k: sorted(v) for k, v in sorted(g.items())}


# --------------------------------------------------------------------------- #
# team snapshots + arbitrary-pairing feature rows (reused idea from multi-step)
# --------------------------------------------------------------------------- #
def snapshots(mf, teams, before):
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


def _row(si, sj, cols, med, host_side):
    """Feature row for i (home) vs j (away). host_side in {None,'i','j'} adds Elo home adv."""
    bi = HOME_ADV if host_side == "i" else 0.0
    bj = HOME_ADV if host_side == "j" else 0.0
    ei, ej = si["elo"] + bi, sj["elo"] + bj
    out = []
    for c in cols:
        if c == "neutral":
            v = 0.0 if host_side else 1.0
        elif c == "year":
            v = 2026.0
        elif c in ("is_world_cup", "is_competitive"):
            v = 1.0
        elif c == "tournament_importance":
            v = 60.0
        elif c == "is_knockout":
            v = 0.0
        elif c == "home_elo_pre":
            v = ei
        elif c == "away_elo_pre":
            v = ej
        elif c in ("elo_diff", "elo_diff_eff"):
            v = ei - ej
        elif c == "elo_sum":
            v = ei + ej
        elif c == "elo_winprob":
            v = 1 / (1 + 10 ** (-(ei - ej) / 400))
        elif c == "is_home_country_home":
            v = 1.0 if host_side == "i" else 0.0
        elif c == "is_home_country_away":
            v = 1.0 if host_side == "j" else 0.0
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


def proba_pairs(pipe, snap, pairs, cols, med):
    """P(H/D/A) for a list of (i, j, host_side) pairings, in one batch."""
    X = np.array([_row(snap[i], snap[j], cols, med, hs) for i, j, hs in pairs], float)
    p = M.proba_hda(pipe, X)
    return {(i, j): p[k] for k, (i, j, _) in enumerate(pairs)}


# --------------------------------------------------------------------------- #
# 32-team strength-reseeded bracket
# --------------------------------------------------------------------------- #
def _seed_order(n):
    seeds = [1, 2]
    while len(seeds) < n:
        m = len(seeds) * 2 + 1
        seeds = [x for s in seeds for x in (s, m - s)]
    return [s - 1 for s in seeds]


SEED32 = _seed_order(32)


def _simulate(groups, fx_rows, Pg, Pn, elo, rng, conditional):
    pts = {t: 0 for g in groups.values() for t in g}
    for r in fx_rows:
        h, a = r["home"], r["away"]
        if conditional and r["played"]:
            res = r["result"]
        else:
            ph, pd, pa = Pg[(h, a)]
            u = rng.random()
            res = 0 if u < ph else (1 if u < ph + pd else 2)
        if res == 0:
            pts[h] += 3
        elif res == 1:
            pts[h] += 1; pts[a] += 1
        else:
            pts[a] += 3
    firsts, seconds, thirds = [], [], []
    for teams in groups.values():
        rk = sorted(teams, key=lambda t: (pts[t], elo.get(t, 1500)), reverse=True)
        firsts.append(rk[0]); seconds.append(rk[1]); thirds.append(rk[2])
    best8 = sorted(thirds, key=lambda t: (pts[t], elo.get(t, 1500)), reverse=True)[:8]
    qualified = firsts + seconds + best8
    ranked = sorted(qualified, key=lambda t: elo.get(t, 1500), reverse=True)   # reseed by strength
    bracket = [ranked[s] for s in SEED32]

    def pwin(a, b):
        ph, pd, pa = Pn[(a, b)]
        return ph + 0.5 * pd

    teams, sf_losers, final_loser = bracket, [], None
    while len(teams) > 1:
        nxt, los = [], []
        for k in range(0, len(teams), 2):
            a, b = teams[k], teams[k + 1]
            w, l = (a, b) if rng.random() < pwin(a, b) else (b, a)
            nxt.append(w); los.append(l)
        if len(teams) == 4:
            sf_losers = los
        if len(teams) == 2:
            final_loser = los[0]
        teams = nxt
    champ = teams[0]
    third = sf_losers[0] if rng.random() < pwin(sf_losers[0], sf_losers[1]) else sf_losers[1]
    return champ, final_loser, third


# --------------------------------------------------------------------------- #
def _train(mf, cols):
    cut = KICKOFF
    tr = (mf.filter(pl.col("date") < cut).filter(~pl.col("is_2026"))
          .filter(pl.col("home_score").is_not_null())
          .filter(pl.col("date") >= cut.replace(year=cut.year - TRAIN_YEARS)).sort("date"))
    y = np.where(tr["home_score"].to_numpy() > tr["away_score"].to_numpy(), 0,
                 np.where(tr["home_score"].to_numpy() == tr["away_score"].to_numpy(), 1, 2))
    med_arr = np.nan_to_num(np.nanmedian(M.to_numpy(tr, cols), axis=0), nan=0.0)
    X = np.where(np.isnan(M.to_numpy(tr, cols)), med_arr, M.to_numpy(tr, cols))
    k = int(len(y) * 0.85)
    for name in MODEL_PRIORITY:
        if name not in M.available_classifiers():
            continue
        try:
            Xf, yf = (X[:k], y[:k]) if name in ("TabPFN", "TabICL") and len(y) > 10000 else (X, y)
            pipe = M.build_classifier(name).fit(Xf, yf)
            cals = M.fit_calibrators(M.proba_hda(pipe, X[k:]), y[k:])
            print(f"model: {name}  (train n={len(yf)}, {cut.year-TRAIN_YEARS}-{cut.year})")
            return name, pipe, cals, dict(zip(cols, med_arr))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip {name}] {type(exc).__name__}: {str(exc)[:70]}")
    raise RuntimeError("no usable classifier")


def _calibrated(p_raw, cals):
    return M.apply_calibrators(p_raw, cals)


def run(mode, groups, fx, snap, cols, pipe, cals, med, elo):
    teams = sorted({t for g in groups.values() for t in g})
    # group-fixture probabilities (host-aware) + neutral knockout matrix
    gpairs = [(r["home"], r["away"], ("i" if r["host"] == r["home"]
                                      else "j" if r["host"] == r["away"] else None))
              for r in fx.iter_rows(named=True)]
    Pg_raw = proba_pairs(pipe, snap, gpairs, cols, med)
    Pg = {k: _calibrated(v[None, :], cals)[0] for k, v in Pg_raw.items()}
    npairs = [(i, j, None) for i in teams for j in teams if i != j]
    Pn_raw = proba_pairs(pipe, snap, npairs, cols, med)
    Pn = {k: _calibrated(v[None, :], cals)[0] for k, v in Pn_raw.items()}

    fx_rows = fx.to_dicts()
    rng = np.random.default_rng(SEED)
    champ, final, semi = {}, {}, {}
    cond = (mode == "conditional")
    for _ in range(N_SIMS):
        c, r, t = _simulate(groups, fx_rows, Pg, Pn, elo, rng, cond)
        for team in (c, r):
            final[team] = final.get(team, 0) + 1
        champ[c] = champ.get(c, 0) + 1
        for team in (c, r, t):
            semi[team] = semi.get(team, 0) + 1     # at least 3rd place == reached SF stage podium
    tab = pl.DataFrame([{"team": t, "P_champion": champ.get(t, 0) / N_SIMS,
                         "P_final": final.get(t, 0) / N_SIMS, "P_podium": semi.get(t, 0) / N_SIMS}
                        for t in teams]).sort("P_champion", descending=True)
    return tab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fx = load_fixtures()
    groups = groups_of(fx)
    teams = sorted({t for g in groups.values() for t in g})
    print(f"loaded {fx.height} fixtures, {len(groups)} groups, {len(teams)} teams, "
          f"{fx['played'].sum()} already played")

    mf = (features.build_match_features_cached(train_cut=dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first"))
    cols = M.feature_columns(mf)
    snap = snapshots(mf, teams, KICKOFF)
    missing = [t for t in teams if t not in snap]
    if missing:
        print(f"[warn] no feature snapshot for: {missing} -> using median features + Elo 1500")
        med0 = {b: None for b in features.TEAM_FEATS}
        for t in missing:
            snap[t] = {**med0, "elo": 1500.0}
    elo = {r["team"]: r["rating"] for r in
           E.latest_ratings().with_columns(canon_team("team").alias("team"))
           .select("team", "rating").iter_rows(named=True)}
    for t in teams:                                # keep snapshot Elo and ratings dict consistent
        elo.setdefault(t, snap[t].get("elo", 1500.0))

    name, pipe, cals, med = _train(mf, cols)
    for mode in ("pretournament", "conditional"):
        tab = run(mode, groups, fx, snap, cols, pipe, cals, med, elo)
        tab.write_csv(OUT / f"wc2026_{mode}.csv")
        print(f"\n=== {mode.upper()} — top 12 (model {name}, {N_SIMS} sims) ===")
        with pl.Config(tbl_rows=12, fmt_float="full"):
            print(tab.with_columns(pl.col("^P_.*$").round(3)).head(12))
    print(f"\nsaved: production/outputs/wc2026_{{pretournament,conditional}}.csv")


if __name__ == "__main__":
    main()
