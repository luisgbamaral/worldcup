"""Loaders return clean, typed Polars frames with the EDA fixes applied."""
import polars as pl

from worldcup import data


def test_results_is_polars(results):
    assert isinstance(results, pl.DataFrame)
    assert results.height > 49_000


def test_played_has_no_null_scores(played):
    assert played["home_score"].null_count() == 0
    assert {"total_goals", "goal_diff", "year"} <= set(played.columns)


def test_goalscorers_bool_flags(goals):
    assert goals.schema["penalty"] == pl.Boolean
    assert goals.schema["own_goal"] == pl.Boolean


def test_former_names_dates_parsed():
    assert data.load_former_names().schema["start_date"] == pl.Date


def test_elo_mixed_dates_all_parsed(elo):
    assert elo["date"].null_count() == 0
    assert elo["date"].max().year >= 2025


def test_latest_elo_one_row_per_team(latest):
    assert latest["team"].n_unique() == latest.height


def test_team_match_log_doubles_rows(played):
    assert data.team_match_log(played).height == 2 * played.height


def test_worldcup_editions_span(editions):
    assert editions["year"].min() == 1930
    assert editions["year"].max() >= 2026


def test_wc2026_loaders():
    assert data.load_squad_players().height == 1248
    assert data.load_squad_coaches().height == 48
    assert data.load_fixtures().schema["match_date"] == pl.Date
