"""Build the clean analytical bases (Parquet) with Polars.

Two outputs in ``data/processed/``:

* ``teams.parquet``   — one row per 2026 World Cup national team (48), with
  squad aggregates, coach, group, latest Elo and all-time results record.
* ``players.parquet`` — one row per squad player (1248), typed and enriched
  with career international goals.

Team names are canonicalized to the **squad-list** spelling across every source
(the Elo file even separates words with non-breaking spaces), so all joins line up.
"""
from __future__ import annotations

import re
import unicodedata

import polars as pl

from . import config


def _norm_name(s: str | None) -> str:
    """Accent-fold + lowercase + collapse spaces, for fuzzy player-name joins."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower().strip())

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
    """Canonicalize a team column: collapse any whitespace (incl. non-breaking
    spaces) to single spaces, trim, then map known aliases."""
    return (pl.col(col).str.replace_all(r"\s+", " ")
            .str.strip_chars().replace(TEAM_ALIASES))


def _read(path) -> pl.DataFrame:
    return pl.read_csv(path, null_values=["NA", ""], infer_schema_length=5000)


def _latest_elo() -> pl.DataFrame:
    """Latest Elo per team (+ world rank), excluding long-dissolved nations."""
    elo = _read(config.ELO_CSV).with_columns(
        canon_team("team").alias("team"),
        pl.coalesce(pl.col("date").str.to_date("%Y-%m-%d", strict=False),
                    pl.col("date").str.to_date("%m/%d/%Y", strict=False)).alias("date"))
    last = (elo.sort("date").group_by("team", maintain_order=True)
            .agg(pl.col("rating").last().alias("elo_rating"),
                 pl.col("date").last().alias("elo_date")))
    cutoff = last["elo_date"].max().replace(year=last["elo_date"].max().year - 1)
    last = last.filter(pl.col("elo_date") >= cutoff)
    return last.with_columns(
        pl.col("elo_rating").rank("dense", descending=True).cast(pl.Int32).alias("elo_rank")
    ).select("team", "elo_rating", "elo_rank")


def _team_history() -> pl.DataFrame:
    """All-time played-match record per team from martj42 results."""
    r = _read(config.RESULTS_CSV).filter(pl.col("home_score").is_not_null())
    home = r.select(canon_team("home_team").alias("team"),
                    pl.col("home_score").alias("gf"), pl.col("away_score").alias("ga"))
    away = r.select(canon_team("away_team").alias("team"),
                    pl.col("away_score").alias("gf"), pl.col("home_score").alias("ga"))
    return pl.concat([home, away]).group_by("team").agg(
        pl.len().alias("matches"),
        ((pl.col("gf") > pl.col("ga")).mean() * 100).round(1).alias("win_pct"),
        pl.col("gf").mean().round(2).alias("gf_per_game"),
        pl.col("ga").mean().round(2).alias("ga_per_game"))


def build_players() -> pl.DataFrame:
    """One typed row per squad player.

    ``goals`` is the squad list's official international tally; ``goals_martj42``
    cross-checks it against the martj42 goal log via accent-folded name match
    (names there read "Given Family"; the squad uses "FAMILY Given").
    """
    key = pl.col("name_key").map_elements(_norm_name, return_dtype=pl.String)
    goals = (_read(config.GOALSCORERS_CSV).filter(~pl.col("own_goal"))
             .select(pl.col("scorer").alias("name_key"))
             .with_columns(key.alias("name_key"))
             .group_by("name_key").agg(pl.len().alias("goals_martj42")))
    p = _read(config.RAW / "squadlist" / "wc2026_squads_players.csv").with_columns(
        canon_team("team").alias("team"),
        pl.col("dob_iso").str.to_date(strict=False).alias("dob"),
        (pl.col("first_names").str.split(" ").list.first()
         + " " + pl.col("last_names")).alias("name_key"))
    return (p.with_columns(key.alias("name_key"))
            .join(goals, on="name_key", how="left")
            .with_columns(
                pl.col("goals_martj42").fill_null(0),
                (canon_team("club_country") != pl.col("team")).alias("plays_abroad"))
            .select("team", "team_code", "number", "position", "player_name",
                    "name_on_shirt", "dob", "age_at_tournament", "height_cm",
                    "caps", "goals", "goals_martj42", "club", "club_country", "plays_abroad")
            .sort("team", "number"))


def build_teams(players: pl.DataFrame) -> pl.DataFrame:
    """One row per 2026 World Cup team: squad aggregates + coach, group, Elo, history."""
    agg = players.group_by("team", "team_code").agg(
        pl.len().alias("squad_size"),
        pl.col("age_at_tournament").mean().round(1).alias("avg_age"),
        pl.col("height_cm").mean().round(1).alias("avg_height_cm"),
        pl.col("caps").sum().alias("total_caps"),
        pl.col("goals").sum().alias("total_goals"),
        pl.col("plays_abroad").sum().alias("n_players_abroad"),
        pl.col("club").n_unique().alias("n_clubs"))

    coaches = _read(config.RAW / "squadlist" / "wc2026_squads_coaches.csv").select(
        canon_team("team").alias("team"), pl.col("coach_name").alias("coach"))

    fx = _read(config.RAW / "fixtures" / "wc2026_group_stage_schedule.csv")
    groups = (pl.concat([fx.select(canon_team("home_team").alias("team"), "group"),
                         fx.select(canon_team("away_team").alias("team"), "group")])
              .group_by("team").agg(pl.col("group").first()))

    return (agg.join(coaches, on="team", how="left")
            .join(groups, on="team", how="left")
            .join(_latest_elo(), on="team", how="left")
            .join(_team_history(), on="team", how="left")
            .sort("elo_rating", descending=True, nulls_last=True))


def main() -> int:
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    players = build_players()
    teams = build_teams(players)
    players.write_parquet(config.PROCESSED / "players.parquet")
    teams.write_parquet(config.PROCESSED / "teams.parquet")
    matched = (players["goals_martj42"] > 0).sum()
    print(f"players.parquet: {players.height} rows x {players.width} cols "
          f"({matched} matched to career goals)")
    print(f"teams.parquet:   {teams.height} rows x {teams.width} cols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
