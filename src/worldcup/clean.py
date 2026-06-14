"""Build the clean analytical bases (Parquet) from the raw loaders.

Two outputs in ``data/processed/``:

* ``teams.parquet``   — one row per 2026 World Cup national team (48): squad
  aggregates, coach, group, latest Elo + rank, all-time results record.
* ``players.parquet`` — one row per squad player (1248), typed and enriched
  with career international goals.

I/O lives in :mod:`worldcup.data`; this module only canonicalizes names and
joins. Team names are standardized to the **squad-list** spelling.
"""
from __future__ import annotations

import re
import unicodedata

import polars as pl

from . import config, data, elo

# Canonical national-team names follow the squad-list spelling; every other
# source (martj42, Elo, fixtures) is mapped onto it.
TEAM_ALIASES = {
    "Bosnia and Herzegovina": "Bosnia And Herzegovina",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Democratic Republic of Congo": "Congo DR",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte D'Ivoire",
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "Turkey": "Türkiye",
    "United States": "USA",
}


def canon_team(col: str) -> pl.Expr:
    """Canonicalize a team column: collapse whitespace (incl. non-breaking
    spaces), trim, then map known aliases to the squad-list spelling."""
    return (pl.col(col).str.replace_all(r"\s+", " ")
            .str.strip_chars().replace(TEAM_ALIASES))


def _norm_name(s: str | None) -> str:
    """Accent-fold + lowercase + collapse spaces, for fuzzy player-name joins."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower().strip())


def _name_key(col: str) -> pl.Expr:
    return pl.col(col).map_elements(_norm_name, return_dtype=pl.String)


def build_players() -> pl.DataFrame:
    """One typed row per squad player.

    ``goals`` is the squad list's official tally; ``goals_martj42`` cross-checks
    it against the martj42 goal log via accent-folded name match (names there
    read "Given Family"; the squad uses "FAMILY Given"). The martj42 log is
    partial, so ``goals`` remains authoritative.
    """
    goals = (data.load_goalscorers().filter(~pl.col("own_goal"))
             .with_columns(canon_team("team").alias("team"),
                           _name_key("scorer").alias("name_key"))
             .group_by("team", "name_key").agg(goals_martj42=pl.len()))
    return (data.load_squad_players()
            .with_columns(
                canon_team("team").alias("team"),
                (pl.col("first_names").str.split(" ").list.first()
                 + " " + pl.col("last_names")).alias("_raw_key"))
            .with_columns(_name_key("_raw_key").alias("name_key"))
            .join(goals, on=["team", "name_key"], how="left")
            .with_columns(
                pl.col("goals_martj42").fill_null(0),
                (canon_team("club_country") != pl.col("team")).alias("plays_abroad"))
            .select("team", "team_code", "number", "position", "player_name",
                    "name_on_shirt", "dob", "age_at_tournament", "height_cm",
                    "caps", "goals", "goals_martj42", "club", "club_country", "plays_abroad")
            .sort("team", "number"))


def _latest_elo() -> pl.DataFrame:
    """Latest self-computed Elo per team (+ world rank), canonical spelling."""
    return (elo.latest_ratings()
            .with_columns(canon_team("team").alias("team"),
                          pl.col("rating").rank("dense", descending=True)
                          .cast(pl.Int32).alias("elo_rank"))
            .select("team", pl.col("rating").alias("elo_rating"), "elo_rank"))


def _team_history() -> pl.DataFrame:
    """All-time played-match record per team (canonical names)."""
    log = (data.team_match_log(data.load_results(played_only=True))
           .with_columns(canon_team("team").alias("team")))
    return log.group_by("team").agg(
        matches=pl.len(),
        win_pct=(pl.col("win").mean() * 100).round(1),
        gf_per_game=pl.col("gf").mean().round(2),
        ga_per_game=pl.col("ga").mean().round(2))


def build_teams(players: pl.DataFrame) -> pl.DataFrame:
    """One row per 2026 World Cup team: squad aggregates + coach, group, Elo, history."""
    agg = players.group_by("team", "team_code").agg(
        squad_size=pl.len(),
        avg_age=pl.col("age_at_tournament").mean().round(1),
        avg_height_cm=pl.col("height_cm").mean().round(1),
        total_caps=pl.col("caps").sum(),
        total_goals=pl.col("goals").sum(),
        n_players_abroad=pl.col("plays_abroad").sum(),
        n_clubs=pl.col("club").n_unique())

    coaches = data.load_squad_coaches().select(
        canon_team("team").alias("team"), pl.col("coach_name").alias("coach"))

    fx = data.load_fixtures()
    groups = (pl.concat([fx.select(canon_team("home_team").alias("team"), "group"),
                         fx.select(canon_team("away_team").alias("team"), "group")])
              .group_by("team").agg(pl.col("group").first()))

    return (agg.join(coaches, on="team", how="left")
            .join(groups, on="team", how="left")
            .join(_latest_elo(), on="team", how="left")
            .join(_team_history(), on="team", how="left")
            .sort("elo_rating", descending=True, nulls_last=True))


def build_matches_rated() -> pl.DataFrame:
    """One row per played match with pre/post self-computed Elo (walk-forward ready)."""
    return elo.rate_matches().select(
        "date", "home_team", "away_team", "home_score", "away_score", "tournament",
        "neutral", "home_elo_pre", "away_elo_pre", "home_elo_post", "away_elo_post")


def main() -> int:
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    players = build_players()
    teams = build_teams(players)
    matches = build_matches_rated()
    players.write_parquet(config.PROCESSED / "players.parquet")
    teams.write_parquet(config.PROCESSED / "teams.parquet")
    matches.write_parquet(config.PROCESSED / "matches_rated.parquet")
    matched = (players["goals_martj42"] > 0).sum()
    print(f"players.parquet:       {players.height} rows x {players.width} cols "
          f"({matched} matched to career goals)")
    print(f"teams.parquet:         {teams.height} rows x {teams.width} cols")
    print(f"matches_rated.parquet: {matches.height} rows x {matches.width} cols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
