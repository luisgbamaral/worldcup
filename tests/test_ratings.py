"""Three rating systems: leakage-safe pre-match ratings + calibrated readouts."""
import datetime as dt

import numpy as np
import polars as pl

from worldcup import data, ratings as R


def _toy():
    return pl.DataFrame({
        "date": [dt.date(2020, 1, 1), dt.date(2020, 1, 8), dt.date(2020, 1, 15)],
        "home_team": ["A", "B", "A"], "away_team": ["B", "A", "C"],
        "home_score": [2, 0, 1], "away_score": [0, 1, 1],
        "neutral": [False, False, True]})


def test_first_match_pre_is_init():
    for fn, init in [(R.rate_classic, 1500.0), (R.rate_pi, 0.0)]:
        d = fn(_toy())
        assert d["home_rating_pre"][0] == init and d["away_rating_pre"][0] == init


def test_elo_classic_equal_and_opposite():
    d = R.rate_classic(_toy(), home_adv=0.0)
    a_pre, b_pre = d["away_rating_pre"][1], d["home_rating_pre"][1]   # match 2: B(home) vs A(away)
    assert abs((a_pre - 1500) + (b_pre - 1500)) < 1e-9               # equal and opposite update


def test_pi_two_finite_ratings():
    d = R.rate_pi(_toy())
    assert d["home_rating_pre"].is_finite().all() and d["away_rating_pre"].is_finite().all()


def test_ratings_leakage_safe():
    # a rating column at row i must not depend on row i's own result: the first
    # appearance of every team is its init value.
    d = R.rate_classic(_toy())
    assert d["home_rating_pre"][0] == 1500 and d["away_rating_pre"][2] == 1500  # C's first match


def test_readout_returns_simplex():
    d = R.rate_world(data.load_results(played_only=True)).filter(pl.col("date").dt.year() >= 2018)
    tr = d.filter(pl.col("date").dt.year() <= 2020)
    va = d.filter(pl.col("date").dt.year() == 2021)
    te = d.filter(pl.col("date").dt.year() >= 2022)
    p = R.Readout().fit(tr, va).predict(te)
    assert np.allclose(p.sum(1), 1) and (p >= 0).all() and (p <= 1).all()
