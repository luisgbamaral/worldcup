"""Monte-Carlo simulator of the 2026 World Cup to the champion (Elo–Poisson engine).

The bracket comes from openfootball (`2026/worldcup.json`): Round-of-32 slots are
group positions (``1E``, ``2A``, ``3A/B/C/D/F`` for a best-third) and later rounds
reference earlier winners (``W74``). Group results known so far come from the
fixtures file; unplayed group games and the whole knockout are simulated.

Match model (any pairing, generalises beyond scheduled games): a Poisson GLM of
goals on the Elo difference, fitted once on history; goals are sampled
independently per team. Knockout draws are decided by an Elo penalty coin-flip.
Best-third → slot allocation is a valid (allowed-group) matching, not the exact
FIFA table (documented approximation).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import polars as pl

from worldcup import config, data, elo as elo_mod, modeling as M
from worldcup.clean import canon_team

ROUND_NAMES = {"Round of 32": "R32", "Round of 16": "R16", "Quarter-final": "QF",
               "Semi-final": "SF", "Final": "Final"}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_bracket() -> list[dict]:
    """Knockout matches (num, round, t1_ref, t2_ref), sorted by match number."""
    ms = json.loads((config.WORLDCUP_JSON_DIR / "2026" / "worldcup.json")
                    .read_text(encoding="utf-8"))["matches"]
    ko = [{"num": m["num"], "round": ROUND_NAMES.get(m["round"]),
           "t1": m["team1"], "t2": m["team2"]}
          for m in ms if m.get("round") in ROUND_NAMES]
    return sorted(ko, key=lambda m: m["num"])


def load_groups() -> dict[str, list[dict]]:
    """Per group letter: its 6 fixtures with (home, away, optional scores)."""
    fx = data.load_fixtures().with_columns(
        canon_team("home_team").alias("home_team"), canon_team("away_team").alias("away_team"))
    out: dict[str, list[dict]] = {}
    for r in fx.iter_rows(named=True):
        out.setdefault(r["group"], []).append(r)
    return out


# --------------------------------------------------------------------------- #
# Elo–Poisson pairing model
# --------------------------------------------------------------------------- #
@dataclass
class Pairing:
    b0: float
    b_elo: float
    b_home: float

    def lam(self, elo_diff: float, home: float = 0.0) -> float:
        return float(np.exp(self.b0 + self.b_elo * elo_diff + self.b_home * home))


def fit_pairing(tmf: pl.DataFrame) -> Pairing:
    from sklearn.linear_model import PoissonRegressor
    X = tmf.select((pl.col("team_elo_pre") - pl.col("opp_elo_pre")).alias("ed"),
                   pl.col("is_home").cast(pl.Float64)).to_numpy()
    y = tmf["gf"].to_numpy().astype(float)
    m = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(X, y)
    return Pairing(float(m.intercept_), float(m.coef_[0]), float(m.coef_[1]))


# --------------------------------------------------------------------------- #
# one simulation
# --------------------------------------------------------------------------- #
def _sample_goals(elo, a, b, pair, rng, home=0.0):
    la = pair.lam(elo[a] - elo[b], home)
    lb = pair.lam(elo[b] - elo[a], 0.0)
    return rng.poisson(la), rng.poisson(lb)


def _standings(group, elo, pair, rng):
    """Final group table as a list of (team, pts, gd, gf), best first."""
    pts = {}; gd = {}; gf = {}
    for fx in group:
        h, a = fx["home_team"], fx["away_team"]
        if fx["home_score"] is not None:
            hs, as_ = fx["home_score"], fx["away_score"]
        else:
            hs, as_ = _sample_goals(elo, h, a, pair, rng)
        for t in (h, a):
            pts.setdefault(t, 0); gd.setdefault(t, 0); gf.setdefault(t, 0)
        pts[h] += 3 if hs > as_ else 1 if hs == as_ else 0
        pts[a] += 3 if as_ > hs else 1 if hs == as_ else 0
        gd[h] += hs - as_; gd[a] += as_ - hs; gf[h] += hs; gf[a] += as_
    return sorted(((t, pts[t], gd[t], gf[t]) for t in pts),
                  key=lambda r: (r[1], r[2], r[3]), reverse=True)


def _assign_thirds(slots, thirds):
    """Match qualified thirds (team, group) to slots (ref, allowed-groups)."""
    assign = {}
    used = set()

    def bt(i):
        if i == len(slots):
            return True
        ref, allowed = slots[i]
        for team, grp in thirds:
            if team not in used and grp in allowed:
                used.add(team); assign[ref] = team
                if bt(i + 1):
                    return True
                used.discard(team); assign.pop(ref, None)
        return False
    bt(0)
    return assign


def simulate_once(elo, groups, bracket, pair, rng):
    table = {g: _standings(groups[g], elo, pair, rng) for g in groups}
    pos = {g: [r[0] for r in table[g]] for g in groups}              # [1st,2nd,3rd,4th]
    # 8 best thirds, ranked by (pts, gd, gf) of each group's third-placed team
    thirds = sorted(((table[g][2][0], g, table[g][2][1], table[g][2][2], table[g][2][3])
                     for g in groups), key=lambda x: (x[2], x[3], x[4]), reverse=True)[:8]
    thirds = [(t, g) for t, g, *_ in thirds]

    slots = []
    for m in bracket:
        for ref in (m["t1"], m["t2"]):
            if "/" in ref:
                slots.append((ref, set(ref[1:].split("/"))))
    third_assign = _assign_thirds(slots, thirds)

    def resolve(ref, winners):
        if ref.startswith("W"):
            return winners[int(ref[1:])]
        if "/" in ref:
            return third_assign.get(ref)
        return pos[ref[1]][int(ref[0]) - 1]

    qualified = {pos[g][0] for g in groups} | {pos[g][1] for g in groups} | set(third_assign.values())
    reached = {t: "R32" for t in qualified}
    winners = {}
    for m in bracket:
        a, b = resolve(m["t1"], winners), resolve(m["t2"], winners)
        if a is None or b is None:
            return reached, None
        reached[a] = reached[b] = m["round"]
        ga, gb = _sample_goals(elo, a, b, pair, rng)
        if ga > gb:
            w = a
        elif gb > ga:
            w = b
        else:  # penalties — Elo coin-flip
            pa = 1 / (1 + 10 ** (-(elo[a] - elo[b]) / 400))
            w = a if rng.random() < pa else b
        winners[m["num"]] = w
    champ = winners[max(winners)]
    reached[champ] = "Champion"
    return reached, champ


# --------------------------------------------------------------------------- #
# Monte-Carlo
# --------------------------------------------------------------------------- #
_ORDER = ["R32", "R16", "QF", "SF", "Final", "Champion"]
_RANK = {r: i for i, r in enumerate(_ORDER)}


def current_elo() -> dict[str, float]:
    """Latest self-computed Elo per team (canonical names) — updates as results arrive."""
    return {r["team"]: r["rating"]
            for r in elo_mod.latest_ratings().with_columns(canon_team("team").alias("team"))
            .select("team", "rating").iter_rows(named=True)}


def load_pairing() -> Pairing:
    tmf = (pl.read_parquet(config.TEAM_MATCH_FEATURES)
           .unique(subset=["match_id", "is_home"], keep="first"))
    return fit_pairing(tmf.filter(pl.col("gf").is_not_null()))


def monte_carlo(n_sims: int = 10000, seed: int = M.SEED) -> pl.DataFrame:
    elo = current_elo()
    groups = load_groups()
    bracket = load_bracket()
    pair = load_pairing()

    teams = sorted({t for g in groups.values() for fx in g for t in (fx["home_team"], fx["away_team"])})
    counts = {t: {r: 0 for r in _ORDER} for t in teams}
    rng = np.random.default_rng(seed)
    for _ in range(n_sims):
        reached, _ = simulate_once(elo, groups, bracket, pair, rng)
        for t, r in reached.items():
            for rr in _ORDER[:_RANK[r] + 1]:
                counts[t][rr] += 1

    rows = [{"team": t, "elo": round(elo.get(t, float("nan")), 1),
             "p_advance": counts[t]["R32"] / n_sims, "p_R16": counts[t]["R16"] / n_sims,
             "p_QF": counts[t]["QF"] / n_sims, "p_SF": counts[t]["SF"] / n_sims,
             "p_final": counts[t]["Final"] / n_sims, "p_champion": counts[t]["Champion"] / n_sims}
            for t in teams]
    return pl.DataFrame(rows).sort("p_champion", descending=True)
