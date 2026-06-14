"""Self-computed Elo: mechanics and agreement with the external source."""
import polars as pl

from worldcup import elo


def test_rate_matches_adds_pre_post_columns(played):
    rated = elo.rate_matches(played)
    assert {"home_elo_pre", "away_elo_pre", "home_elo_post", "away_elo_post"} <= set(rated.columns)
    assert rated.height == played.height


def test_latest_ratings_one_row_per_team():
    latest = elo.latest_ratings()
    assert latest["team"].n_unique() == latest.height


def test_agreement_with_eloratings_net():
    cmp = elo.compare_to_external()
    # ranking must track the established source closely for the baseline to be valid
    rank_corr = cmp.select(pl.corr(pl.col("elo_ours").rank(),
                                   pl.col("elo_orig").rank())).item()
    assert rank_corr > 0.9
