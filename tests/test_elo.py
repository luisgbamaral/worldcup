"""Self-computed Elo mechanics and sanity."""
import polars as pl

from worldcup import elo


def test_rate_matches_adds_pre_post_columns(rated, played):
    assert {"home_elo_pre", "away_elo_pre", "home_elo_post", "away_elo_post"} <= set(rated.columns)
    assert rated.height == played.height


def test_latest_ratings_one_row_per_team(latest):
    assert latest["team"].n_unique() == latest.height
    assert latest["rating"].is_not_null().all()


def test_strong_teams_rank_on_top(latest):
    # face validity: traditional powers should sit near the top
    top10 = set(latest.head(10)["team"].to_list())
    assert {"Brazil", "Spain", "Argentina"} & top10


def test_zero_sum_updates(played):
    # each match moves the two teams by equal and opposite amounts
    rated = elo.rate_matches(played).with_columns(
        (pl.col("home_elo_post") - pl.col("home_elo_pre")).alias("dh"),
        (pl.col("away_elo_post") - pl.col("away_elo_pre")).alias("da"))
    assert (rated.select((pl.col("dh") + pl.col("da")).abs().max()).item()) < 1e-9
