"""Consolidated feature base: leakage, cut, fixtures, differentials, squad, canon."""
import datetime as dt

import polars as pl
import pytest

from worldcup import clean, data, features


@pytest.fixture(scope="module")
def mf():
    return features.build_match_features()


def test_no_leakage(played):
    # synthetic team with a known scoring sequence proves the shift excludes the
    # current match and uses only strictly-earlier games.
    df = pl.DataFrame({
        "team": ["X"] * 6,
        "date": [dt.date(2020, 1, d) for d in range(1, 7)],
        "gf": [1, 2, 3, 4, 5, 6], "ga": [0] * 6,
        "opp_elo_pre": [1500.0] * 6, "elo_pre": [1500.0] * 6, "elo_post": [1500.0] * 6,
    }).sort("team", "date")
    out = features._add_form(df, 1)
    assert out["gf_pg_5"][0] is None                 # no prior match
    assert out["gf_pg_5"][5] == pytest.approx(3.0)   # mean of the 5 strictly-earlier gf


def test_cut(mf):
    hist = mf.filter(~pl.col("is_2026"))
    assert hist["date"].min() >= dt.date(2018, 1, 1)
    assert mf.filter(pl.col("is_2026")).height == 72   # 2026 present regardless of date


def test_fixtures_present(mf):
    fx = mf.filter(pl.col("is_2026"))
    assert fx.height == 72
    assert fx["played"].sum() == 8
    assert (~fx["played"]).sum() == 64


def test_differentials(mf):
    for c in ("ppg_5", "gd_pg_10", "squad_avg_age"):
        d = mf.select((pl.col(f"{c}_diff")
                       - (pl.col(f"{c}_home") - pl.col(f"{c}_away"))).abs().max()).item()
        assert d is None or d == pytest.approx(0.0, abs=1e-9)
    assert mf.select((pl.col("elo_diff")
                      - (pl.col("home_elo_pre") - pl.col("away_elo_pre"))).abs().max()).item() == 0.0


def test_targets_raw(mf):
    raw_max = (data.load_results(played_only=True)
               .filter(pl.col("date") >= dt.date(2018, 1, 1))["total_goals"].max())
    assert mf.filter(pl.col("played"))["total_goals"].max() == raw_max  # never clipped


def test_squad_null_hist(mf):
    hist, fx = mf.filter(~pl.col("is_2026")), mf.filter(pl.col("is_2026"))
    for c in ("squad_avg_age_home", "share_top5_league_away", "star_caps_diff"):
        assert hist[c].null_count() == hist.height
        assert fx[c].null_count() == 0


def test_canon(mf):
    teams48 = set(pl.read_parquet(features.config.PROCESSED / "teams.parquet")["team"])
    fx = mf.filter(pl.col("is_2026"))
    names = set(fx["home_team"]) | set(fx["away_team"])
    assert names <= teams48
    aliases = set(clean.TEAM_ALIASES)               # martj42 spellings that must be mapped
    allnames = set(mf["home_team"]) | set(mf["away_team"])
    assert not (allnames & aliases)
