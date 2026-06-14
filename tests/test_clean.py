"""Name canonicalization and the two clean Parquet bases."""
import polars as pl

from worldcup import clean

NBSP = chr(0xA0)  # the separator the Elo source uses between words


def test_canon_team_maps_to_squad_spelling():
    out = (pl.DataFrame({"t": ["Czech Republic", "Turkey", "United States", "Brazil"]})
           .select(clean.canon_team("t"))["t"].to_list())
    assert out == ["Czechia", "Türkiye", "USA", "Brazil"]


def test_canon_team_strips_non_breaking_space():
    out = pl.DataFrame({"t": [f"New{NBSP}Zealand"]}).select(clean.canon_team("t"))["t"][0]
    assert out == "New Zealand"


def test_players_base_shape_and_types(players):
    assert (players.height, players.width) == (1248, 15)
    assert players.schema["plays_abroad"] == pl.Boolean
    assert players.schema["dob"] == pl.Date


def test_teams_base_complete(teams):
    assert (teams.height, teams.width) == (48, 17)
    assert teams["elo_rating"].null_count() == 0   # every team joined to Elo
    assert teams["matches"].null_count() == 0       # ...and to its history


def test_goals_martj42_rarely_exceeds_official(players):
    # martj42 is a partial, independent log joined on (team, name); it should
    # almost never exceed the official tally — only rare cross-source
    # discrepancies (not homonyms) remain.
    violations = players.filter(pl.col("goals_martj42") > pl.col("goals")).height
    assert violations / players.height < 0.01
