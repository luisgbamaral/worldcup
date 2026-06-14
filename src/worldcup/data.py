"""Data loaders for the raw football datasets.

Each loader returns a tidy ``pandas.DataFrame`` and encapsulates the
data-quality fixes discovered during EDA, so downstream code never has to
re-handle them:

* Elo CSV mixes two date formats (ISO ``1872-11-30`` and US ``12/13/2025``).
* ``results`` contains future, not-yet-played fixtures (NaN scores).
* Boolean-ish columns are stored as the strings ``"TRUE"``/``"FALSE"``.
* Dissolved nations (e.g. *West Germany*) coexist with current ones.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from . import config


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().eq("TRUE")


def load_results(played_only: bool = False) -> pd.DataFrame:
    """International match results (martj42), 1872 → 2026.

    Parameters
    ----------
    played_only:
        Drop future/unplayed fixtures and cast scores to ``int``.
    """
    df = pd.read_csv(config.RESULTS_CSV, parse_dates=["date"])
    df["neutral"] = _to_bool(df["neutral"])
    if played_only:
        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        df["total_goals"] = df["home_score"] + df["away_score"]
        df["goal_diff"] = df["home_score"] - df["away_score"]
        df["year"] = df["date"].dt.year
    return df


def load_goalscorers() -> pd.DataFrame:
    """Individual goals with scorer, minute, penalty and own-goal flags."""
    df = pd.read_csv(config.GOALSCORERS_CSV, parse_dates=["date"])
    df["own_goal"] = _to_bool(df["own_goal"])
    df["penalty"] = _to_bool(df["penalty"])
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    return df


def load_shootouts() -> pd.DataFrame:
    return pd.read_csv(config.SHOOTOUTS_CSV, parse_dates=["date"])


def load_former_names() -> pd.DataFrame:
    """Mapping of former → current national-team names, with valid window."""
    return pd.read_csv(
        config.FORMER_NAMES_CSV, parse_dates=["start_date", "end_date"]
    )


def load_elo() -> pd.DataFrame:
    """Historical Elo ratings.

    Fixes the mixed date encoding (ISO vs. US) that silently nulls ~99% of
    rows under a single-format parse.
    """
    df = pd.read_csv(config.ELO_CSV)
    iso = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    us = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["date"] = iso.fillna(us)
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def latest_elo(exclude_dissolved: bool = True) -> pd.DataFrame:
    """Most recent Elo snapshot per team.

    ``exclude_dissolved`` drops teams whose last rating predates the latest
    global snapshot (e.g. *West Germany*, *Yugoslavia*), which otherwise
    pollute "current strength" rankings.
    """
    elo = load_elo()
    last = elo.groupby("team", as_index=False).tail(1)
    if exclude_dissolved:
        cutoff = elo["date"].max() - pd.Timedelta(days=365)
        last = last[last["date"] >= cutoff]
    return last.sort_values("rating", ascending=False).reset_index(drop=True)


def team_match_log(results_played: pd.DataFrame) -> pd.DataFrame:
    """Reshape match results to one row per (team, match): gf, ga, outcome."""
    home = results_played.rename(
        columns={"home_team": "team", "away_team": "opponent",
                 "home_score": "gf", "away_score": "ga"}
    )
    away = results_played.rename(
        columns={"away_team": "team", "home_team": "opponent",
                 "away_score": "gf", "home_score": "ga"}
    )
    cols = ["date", "team", "opponent", "gf", "ga", "tournament", "neutral"]
    log = pd.concat([home[cols], away[cols]], ignore_index=True)
    log["win"] = log["gf"] > log["ga"]
    log["draw"] = log["gf"] == log["ga"]
    log["loss"] = log["gf"] < log["ga"]
    return log


def load_worldcup_editions() -> pd.DataFrame:
    """One row per World Cup edition: played matches, goals, goals/match."""
    rows = []
    pattern = str(config.WORLDCUP_JSON_DIR / "*" / "worldcup.json")
    for path in sorted(glob.glob(pattern)):
        year = Path(path).parent.name
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        played, goals = 0, 0
        for m in doc.get("matches", []):
            ft = (m.get("score") or {}).get("ft")
            if ft and len(ft) == 2 and all(isinstance(x, (int, float)) for x in ft):
                played += 1
                goals += ft[0] + ft[1]
        if played:
            rows.append((int(year), played, goals, goals / played))
    return pd.DataFrame(
        rows, columns=["year", "matches", "goals", "goals_per_match"]
    ).sort_values("year").reset_index(drop=True)
