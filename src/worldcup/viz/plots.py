"""EDA figures. Each function returns a Matplotlib ``Figure`` drawn at NeurIPS
size; persist it with :func:`worldcup.viz.save_fig`."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .style import TEXTWIDTH_IN, HALFWIDTH_IN, style_for


def _fig(h_ratio: float = 0.50):
    return plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * h_ratio))


def matches_per_year(played: pd.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    played.groupby("year").size().plot(ax=ax, color=style_for(0)["color"])
    ax.set(title="International matches per year", xlabel="year", ylabel="matches")
    return fig


def goals_per_match_trend(played: pd.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    played.groupby("year")["total_goals"].mean().plot(ax=ax, color=style_for(3)["color"])
    ax.set(title="Average goals per match over time", xlabel="year", ylabel="goals / match")
    return fig


def goals_distribution(played: pd.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    played["total_goals"].clip(upper=10).plot(
        kind="hist", bins=range(0, 12), ax=ax, color=style_for(2)["color"], rwidth=0.9)
    ax.set(title="Goals per match (capped at 10)", xlabel="goals in match", ylabel="frequency")
    return fig


def home_advantage_by_decade(played: pd.DataFrame) -> plt.Figure:
    nz = played[~played["neutral"]].copy()
    nz["decade"] = (nz["year"] // 10) * 10
    g = nz.groupby("decade")
    series = {
        "home win": g.apply(lambda d: (d.home_score > d.away_score).mean()),
        "draw": g.apply(lambda d: (d.home_score == d.away_score).mean()),
        "away win": g.apply(lambda d: (d.home_score < d.away_score).mean()),
    }
    fig, ax = _fig()
    for i, (label, s) in enumerate(series.items()):
        ax.plot(s.index, s.values * 100, label=label, **style_for(i))
    ax.set(title="Outcome by venue (non-neutral) per decade", xlabel="decade", ylabel="%")
    ax.legend()
    return fig


def goal_minute_distribution(goals: pd.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    goals["minute"].dropna().clip(upper=95).plot(
        kind="hist", bins=range(0, 100, 5), ax=ax, color=style_for(4)["color"], rwidth=0.9)
    ax.set(title="Distribution of goal minute", xlabel="minute", ylabel="frequency")
    return fig


def worldcup_goals_per_match(editions: pd.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    ax.plot(editions["year"], editions["goals_per_match"], **style_for(1))
    ax.set(title="Goals per match across World Cups", xlabel="year", ylabel="goals / match")
    return fig


def top_elo(latest: pd.DataFrame, n: int = 12) -> plt.Figure:
    top = latest.head(n).sort_values("rating")
    fig, ax = _fig(0.6)
    ax.barh(top["team"], top["rating"], color=style_for(0)["color"])
    ax.set(title=f"Top {n} teams by current Elo", xlabel="Elo rating")
    return fig


def elo_history(elo: pd.DataFrame, teams: list[str]) -> plt.Figure:
    """Elo trajectory for selected teams (line style per team for B&W)."""
    fig, ax = _fig()
    for i, t in enumerate(teams):
        s = elo[elo["team"] == t].set_index("date")["rating"]
        ax.plot(s.index, s.values, label=t, marker="", **{k: v for k, v in style_for(i).items() if k != "marker"})
    ax.set(title="Elo rating over time", xlabel="year", ylabel="Elo rating")
    ax.legend(ncol=2)
    return fig
