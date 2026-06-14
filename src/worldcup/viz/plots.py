"""EDA figures. Each function takes a Polars DataFrame and returns a Matplotlib
``Figure`` drawn at NeurIPS size; persist it with :func:`worldcup.viz.save_fig`."""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl

from .style import TEXTWIDTH_IN, style_for


def _fig(h_ratio: float = 0.50):
    return plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * h_ratio))


def matches_per_year(played: pl.DataFrame) -> plt.Figure:
    s = played.group_by("year").len().sort("year")
    fig, ax = _fig()
    ax.plot(s["year"], s["len"], color=style_for(0)["color"])
    ax.set(title="International matches per year", xlabel="year", ylabel="matches")
    return fig


def goals_per_match_trend(played: pl.DataFrame) -> plt.Figure:
    s = played.group_by("year").agg(pl.col("total_goals").mean()).sort("year")
    fig, ax = _fig()
    ax.plot(s["year"], s["total_goals"], color=style_for(3)["color"])
    ax.set(title="Average goals per match over time", xlabel="year", ylabel="goals / match")
    return fig


def goals_distribution(played: pl.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    ax.hist(played["total_goals"].clip(upper_bound=10).to_numpy(),
            bins=range(0, 12), color=style_for(2)["color"], rwidth=0.9)
    ax.set(title="Goals per match (capped at 10)", xlabel="goals in match", ylabel="frequency")
    return fig


def home_advantage_by_decade(played: pl.DataFrame) -> plt.Figure:
    res = (played.filter(~pl.col("neutral"))
           .with_columns(((pl.col("year") // 10) * 10).alias("decade"))
           .group_by("decade").agg(
               (pl.col("home_score") > pl.col("away_score")).mean().alias("home win"),
               (pl.col("home_score") == pl.col("away_score")).mean().alias("draw"),
               (pl.col("home_score") < pl.col("away_score")).mean().alias("away win"))
           .sort("decade"))
    fig, ax = _fig()
    for i, label in enumerate(["home win", "draw", "away win"]):
        ax.plot(res["decade"], res[label] * 100, label=label, **style_for(i))
    ax.set(title="Outcome by venue (non-neutral) per decade", xlabel="decade", ylabel="%")
    ax.legend()
    return fig


def goal_minute_distribution(goals: pl.DataFrame) -> plt.Figure:
    m = goals["minute"].drop_nulls().clip(upper_bound=95).to_numpy()
    fig, ax = _fig()
    ax.hist(m, bins=range(0, 100, 5), color=style_for(4)["color"], rwidth=0.9)
    ax.set(title="Distribution of goal minute", xlabel="minute", ylabel="frequency")
    return fig


def worldcup_goals_per_match(editions: pl.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    ax.plot(editions["year"], editions["goals_per_match"], **style_for(1))
    ax.set(title="Goals per match across World Cups", xlabel="year", ylabel="goals / match")
    return fig


def top_elo(latest: pl.DataFrame, n: int = 12) -> plt.Figure:
    top = latest.head(n).sort("rating")
    fig, ax = _fig(0.6)
    ax.barh(top["team"], top["rating"], color=style_for(0)["color"])
    ax.set(title=f"Top {n} teams by current Elo", xlabel="Elo rating")
    return fig


def elo_vs_external(cmp: pl.DataFrame) -> plt.Figure:
    """Scatter of self-computed vs eloratings.net ratings, with the y=x line."""
    lo = min(cmp["elo_ours"].min(), cmp["elo_orig"].min())
    hi = max(cmp["elo_ours"].max(), cmp["elo_orig"].max())
    fig, ax = _fig(0.8)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", label="y = x")
    ax.scatter(cmp["elo_orig"], cmp["elo_ours"], s=8,
               color=style_for(0)["color"], alpha=0.7)
    ax.set(title="Self-computed vs eloratings.net (latest per team)",
           xlabel="eloratings.net rating", ylabel="self-computed rating")
    ax.legend()
    return fig


def elo_history(elo: pl.DataFrame, teams: list[str]) -> plt.Figure:
    """Elo trajectory for selected teams (line style per team for B&W)."""
    fig, ax = _fig()
    for i, t in enumerate(teams):
        s = elo.filter(pl.col("team") == t).sort("date")
        kw = {k: v for k, v in style_for(i).items() if k != "marker"}
        ax.plot(s["date"], s["rating"], label=t, marker="", **kw)
    ax.set(title="Elo rating over time", xlabel="year", ylabel="Elo rating")
    ax.legend(ncol=2)
    return fig
