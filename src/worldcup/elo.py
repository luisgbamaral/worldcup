"""Self-computed World Football Elo ratings from match results.

Implements the eloratings.net update rule, rolled forward over every played
match. Unlike the external snapshot (which stops at 2025-12-13) this gives a
rating *before every game* — what the walk-forward baseline needs — and is
fully reproducible.

    R' = R + K (W - We),  We = 1 / (1 + 10^(-dr/400))

with a home-advantage bonus on ``dr``, an importance weight ``K0`` per
tournament, and a goal-difference multiplier ``G`` (``K = K0 * G``).
"""
from __future__ import annotations

import polars as pl

from . import data

HOME_ADV = 100.0
INIT_RATING = 1500.0


def _k0(tournament: str) -> int:
    """eloratings.net importance weight by tournament type."""
    t = tournament.lower()
    if "friendly" in t:
        return 20
    if "qualif" in t:
        return 40
    if "world cup" in t:
        return 60
    if any(k in t for k in ("uefa euro", "copa am", "african cup", "asian cup",
                            "gold cup", "oceania", "confederations")):
        return 50
    return 30


def _g(goal_diff: int) -> float:
    """Goal-difference multiplier."""
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11 + n) / 8


def rate_matches(played: pl.DataFrame | None = None) -> pl.DataFrame:
    """Append pre/post Elo for both teams to each played match (chronological)."""
    df = (played if played is not None else data.load_results(played_only=True)).sort("date")
    rating: dict[str, float] = {}
    h_pre, a_pre, h_post, a_post = [], [], [], []
    for m in df.iter_rows(named=True):
        h, a = m["home_team"], m["away_team"]
        rh, ra = rating.get(h, INIT_RATING), rating.get(a, INIT_RATING)
        h_pre.append(rh)
        a_pre.append(ra)
        adv = 0.0 if m["neutral"] else HOME_ADV
        we = 1.0 / (1.0 + 10 ** (-((rh + adv) - ra) / 400))
        gd = m["home_score"] - m["away_score"]
        w = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        delta = _k0(m["tournament"]) * _g(gd) * (w - we)
        rating[h] = rh + delta
        rating[a] = ra - delta
        h_post.append(rating[h])
        a_post.append(rating[a])
    return df.with_columns(home_elo_pre=pl.Series(h_pre), away_elo_pre=pl.Series(a_pre),
                           home_elo_post=pl.Series(h_post), away_elo_post=pl.Series(a_post))


def rating_history(rated: pl.DataFrame | None = None) -> pl.DataFrame:
    """Post-match rating per team over time (long): date, team, rating."""
    rated = rated if rated is not None else rate_matches()
    return pl.concat([
        rated.select("date", team="home_team", rating="home_elo_post"),
        rated.select("date", team="away_team", rating="away_elo_post")]).sort("date")


def latest_ratings(rated: pl.DataFrame | None = None) -> pl.DataFrame:
    """Most recent computed rating per team (sorted strongest first)."""
    return (rating_history(rated).group_by("team", maintain_order=True).last()
            .sort("rating", descending=True))
