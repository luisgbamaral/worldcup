"""Data loaders for the raw football datasets — Polars throughout.

Each loader returns a tidy ``polars.DataFrame`` and encapsulates the
data-quality fixes found during EDA, so downstream code never re-handles them:

* Elo CSV mixes two date formats (ISO ``1872-11-30`` and US ``12/13/2025``).
* ``results`` contains future, not-yet-played fixtures (``NA`` scores).
* Dissolved nations (e.g. *West Germany*) coexist with current ones.
"""
from __future__ import annotations

import json

import polars as pl

from . import config


def _read_csv(path) -> pl.DataFrame:
    return pl.read_csv(path, null_values=["NA", ""], infer_schema_length=5000)


def _iso_date(col: str = "date") -> pl.Expr:
    return pl.col(col).str.to_date("%Y-%m-%d", strict=False)


def _mixed_date(col: str = "date") -> pl.Expr:
    return pl.coalesce(pl.col(col).str.to_date("%Y-%m-%d", strict=False),
                       pl.col(col).str.to_date("%m/%d/%Y", strict=False))


def load_results(played_only: bool = False) -> pl.DataFrame:
    """International match results (martj42), 1872 → 2026."""
    df = _read_csv(config.RESULTS_CSV).with_columns(_iso_date().alias("date"))
    if played_only:
        df = df.filter(pl.col("home_score").is_not_null()).with_columns(
            (pl.col("home_score") + pl.col("away_score")).alias("total_goals"),
            (pl.col("home_score") - pl.col("away_score")).alias("goal_diff"),
            pl.col("date").dt.year().alias("year"))
    return df


def load_goalscorers() -> pl.DataFrame:
    """Individual goals with scorer, minute, penalty and own-goal flags."""
    return _read_csv(config.GOALSCORERS_CSV).with_columns(_iso_date().alias("date"))


def load_shootouts() -> pl.DataFrame:
    return _read_csv(config.SHOOTOUTS_CSV).with_columns(_iso_date().alias("date"))


def load_former_names() -> pl.DataFrame:
    """Mapping of former → current national-team names, with valid window."""
    return _read_csv(config.FORMER_NAMES_CSV).with_columns(
        _iso_date("start_date").alias("start_date"),
        _iso_date("end_date").alias("end_date"))


def load_elo() -> pl.DataFrame:
    """Historical Elo ratings (mixed date encoding fixed)."""
    return (_read_csv(config.ELO_CSV)
            .with_columns(_mixed_date().alias("date"))
            .drop_nulls("date").sort("date"))


def latest_elo(exclude_dissolved: bool = True) -> pl.DataFrame:
    """Most recent Elo snapshot per team (sorted by rating).

    ``exclude_dissolved`` drops teams whose last rating predates the latest
    global snapshot by over a year (e.g. *West Germany*).
    """
    elo = load_elo()
    last = elo.group_by("team", maintain_order=True).last()
    if exclude_dissolved:
        cutoff = last["date"].max().replace(year=last["date"].max().year - 1)
        last = last.filter(pl.col("date") >= cutoff)
    return last.sort("rating", descending=True)


def team_match_log(results_played: pl.DataFrame) -> pl.DataFrame:
    """Reshape match results to one row per (team, match): gf, ga, outcome."""
    home = results_played.select(
        "date", pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent"),
        pl.col("home_score").alias("gf"), pl.col("away_score").alias("ga"),
        "tournament", "neutral")
    away = results_played.select(
        "date", pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent"),
        pl.col("away_score").alias("gf"), pl.col("home_score").alias("ga"),
        "tournament", "neutral")
    return pl.concat([home, away]).with_columns(
        (pl.col("gf") > pl.col("ga")).alias("win"),
        (pl.col("gf") == pl.col("ga")).alias("draw"),
        (pl.col("gf") < pl.col("ga")).alias("loss"))


def load_worldcup_editions() -> pl.DataFrame:
    """One row per World Cup edition: played matches, goals, goals/match."""
    rows = []
    for path in sorted(config.WORLDCUP_JSON_DIR.glob("*/worldcup.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        played, goals = 0, 0
        for m in doc.get("matches", []):
            ft = (m.get("score") or {}).get("ft")
            if ft and len(ft) == 2 and all(isinstance(x, (int, float)) for x in ft):
                played += 1
                goals += ft[0] + ft[1]
        if played:
            rows.append({"year": int(path.parent.name), "matches": played,
                         "goals": goals, "goals_per_match": goals / played})
    return pl.DataFrame(rows).sort("year")
